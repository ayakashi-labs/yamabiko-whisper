//! Streaming Whisper processor that drives a [`HypothesisBuffer`].
//!
//! Mirrors the `OnlineASRProcessor` in `ufal/whisper_streaming`: holds a
//! rolling 16 kHz f32 audio buffer, runs Whisper on the whole buffer every
//! `process_iter` call, feeds word-level hypotheses into the
//! LocalAgreement-2 buffer, and trims old audio once committed words pile
//! up past `buffer_trimming_sec`.
//!
//! The state object uses an `Arc<WhisperInnerContext>` internally, so
//! `OnlineAsrProcessor` does not need a lifetime tied to the borrowed
//! `WhisperContext` — the context can be dropped after `new()` returns and
//! the processor remains usable.

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperState};

use crate::hypothesis_buffer::{HypothesisBuffer, Word};

const SAMPLING_RATE: usize = 16_000;
const BUFFER_TRIMMING_SEC: f64 = 15.0;
const PROMPT_CHAR_BUDGET: usize = 200;

/// Streaming wrapper around `whisper-rs` implementing LocalAgreement-2.
pub struct OnlineAsrProcessor {
    state: WhisperState,
    audio_buffer: Vec<f32>,
    buffer_time_offset: f64,
    transcript_buffer: HypothesisBuffer,
    committed: Vec<Word>,
    sep: &'static str,
    language: String,
    n_threads: i32,
    buffer_trimming_sec: f64,
}

impl OnlineAsrProcessor {
    /// Create a processor backed by `ctx`. `language` is the Whisper
    /// language code (`"en"`, `"ja"`, …) or `"auto"` for autodetect.
    pub fn new(ctx: &WhisperContext, language: &str) -> Self {
        let state = ctx.create_state().expect("failed to create whisper state");
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        let sep = if is_cjk(language) { "" } else { " " };
        Self {
            state,
            audio_buffer: Vec::new(),
            buffer_time_offset: 0.0,
            transcript_buffer: HypothesisBuffer::new(),
            committed: Vec::new(),
            sep,
            language: language.to_string(),
            n_threads,
            buffer_trimming_sec: BUFFER_TRIMMING_SEC,
        }
    }

    /// Append 16 kHz mono f32 PCM samples to the rolling buffer.
    pub fn insert_audio_chunk(&mut self, pcm: &[f32]) {
        self.audio_buffer.extend_from_slice(pcm);
    }

    /// Run one Whisper pass over the rolling buffer and return the words
    /// that were just committed by LocalAgreement-2.
    pub fn process_iter(&mut self) -> Vec<Word> {
        if self.audio_buffer.is_empty() {
            return Vec::new();
        }

        let prompt = self.build_init_prompt();

        // Match the reference whisper_streaming defaults (BeamSearch with
        // beam_size=5). Greedy is cheaper but produces less stable
        // predictions across overlapping audio windows, which defeats
        // LocalAgreement-2 (it relies on the same prefix being predicted
        // twice in a row).
        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        });
        if !self.language.is_empty() && self.language != "auto" {
            params.set_language(Some(&self.language));
        }
        params.set_n_threads(self.n_threads);
        // One word per segment: whisper.cpp uses `max_len` as a character
        // budget per segment and `split_on_word` to constrain the cuts
        // to word boundaries. Combined, each segment is exactly one word
        // with stable t0/t1 — far more usable than per-token timestamps
        // which drift across passes (especially in CJK where BPE pieces
        // are sub-character).
        params.set_token_timestamps(true);
        params.set_max_len(1);
        params.set_split_on_word(true);
        // Context comes from `init_prompt`; let whisper.cpp keep KV cache
        // disabled across calls so each pass is reproducible.
        params.set_no_context(true);
        params.set_single_segment(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);
        if !prompt.is_empty() {
            params.set_initial_prompt(&prompt);
        }

        if self.state.full(params, &self.audio_buffer).is_err() {
            return Vec::new();
        }

        let words = self.extract_words();
        self.transcript_buffer
            .insert(words, self.buffer_time_offset);
        let committed = self.transcript_buffer.flush();
        self.committed.extend(committed.iter().cloned());

        self.maybe_trim();

        committed
    }

    /// Drain the still-tentative buffer at end of stream.
    pub fn finish(&mut self) -> Vec<Word> {
        let remaining = self.transcript_buffer.complete();
        self.buffer_time_offset += self.audio_buffer.len() as f64 / SAMPLING_RATE as f64;
        remaining
    }

    /// Snapshot of the words that are predicted but not yet committed.
    /// Used by the example to render the in-flight overlay line.
    pub fn tentative(&self) -> Vec<Word> {
        self.transcript_buffer.complete()
    }

    /// Word separator for this language: `" "` for space-segmented
    /// languages, `""` for CJK.
    pub fn sep(&self) -> &'static str {
        self.sep
    }

    /// Reverse-walk committed words and join the most recent ones
    /// (≤ `PROMPT_CHAR_BUDGET` chars worth) into an init-prompt string,
    /// considering only words that have already left the audio buffer.
    fn build_init_prompt(&self) -> String {
        let older: Vec<&str> = self
            .committed
            .iter()
            .filter(|w| w.end <= self.buffer_time_offset)
            .map(|w| w.text.as_str())
            .collect();
        let mut taken: Vec<&str> = Vec::new();
        let mut chars = 0usize;
        for s in older.iter().rev() {
            let bump = s.chars().count() + 1;
            if chars + bump > PROMPT_CHAR_BUDGET {
                break;
            }
            chars += bump;
            taken.push(s);
        }
        taken.reverse();
        taken.join(self.sep)
    }

    /// Read one Word per Whisper segment, relying on `max_len(1)` +
    /// `split_on_word(true)` having forced each segment to be exactly one
    /// word with proper t0/t1. Timestamps come back relative to the
    /// current audio buffer; the offset is added later inside
    /// [`HypothesisBuffer::insert`].
    fn extract_words(&self) -> Vec<Word> {
        let n_segments = self.state.full_n_segments();
        let mut words: Vec<Word> = Vec::new();
        for seg_idx in 0..n_segments {
            let Some(segment) = self.state.get_segment(seg_idx) else {
                continue;
            };
            let Ok(text_cow) = segment.to_str_lossy() else {
                continue;
            };
            let text = text_cow.trim().to_string();
            if text.is_empty() {
                continue;
            }
            let start = segment.start_timestamp() as f64 / 100.0;
            let mut end = segment.end_timestamp() as f64 / 100.0;
            // Whisper's segment timestamps are reasonable but can briefly
            // emit a zero-width or slightly inverted span — clamp to a
            // minimum positive duration so downstream comparisons stay
            // monotonic.
            if end < start + 0.02 {
                end = start + 0.02;
            }
            words.push(Word { start, end, text });
        }
        words
    }

    /// Trim the audio buffer once it grows past `buffer_trimming_sec`.
    ///
    /// Two cut-point strategies, in order:
    /// 1. End of the last committed word (preferred — fully aligned with
    ///    LocalAgreement state).
    /// 2. End of the second-to-last Whisper segment (fallback — used when
    ///    LocalAgreement has not been firing, e.g. CJK predictions that
    ///    flicker across iterations and never produce a stable prefix).
    ///
    /// Strategy (2) only trims audio; it does NOT promote tentative words
    /// to committed, since those words have not been confirmed by
    /// LocalAgreement and force-committing them would degrade overall
    /// transcription accuracy. The trade-off is that some unconfirmed
    /// hypotheses fall off the back of the buffer when this fires.
    fn maybe_trim(&mut self) {
        let buf_secs = self.audio_buffer.len() as f64 / SAMPLING_RATE as f64;
        if buf_secs <= self.buffer_trimming_sec {
            return;
        }

        let cut_time = self
            .commit_based_cut()
            .or_else(|| self.segment_end_cut())
            .filter(|&t| t > self.buffer_time_offset);
        let Some(cut_time) = cut_time else {
            return;
        };

        self.transcript_buffer.pop_committed(cut_time);
        let cut_samples = (((cut_time - self.buffer_time_offset) * SAMPLING_RATE as f64) as usize)
            .min(self.audio_buffer.len());
        self.audio_buffer.drain(..cut_samples);
        self.buffer_time_offset = cut_time;
        // Keep the start-time filter from rejecting future words that
        // legitimately start near the new buffer origin.
        if self.transcript_buffer.last_committed_time < cut_time {
            self.transcript_buffer.last_committed_time = cut_time;
        }
    }

    fn commit_based_cut(&self) -> Option<f64> {
        self.committed
            .last()
            .map(|w| w.end)
            .filter(|&e| e > self.buffer_time_offset)
    }

    fn segment_end_cut(&self) -> Option<f64> {
        let n = self.state.full_n_segments();
        if n < 2 {
            return None;
        }
        // The second-to-last segment is much more stable than the last,
        // which is often still mid-utterance.
        let seg = self.state.get_segment(n - 2)?;
        let end_sec = seg.end_timestamp() as f64 / 100.0 + self.buffer_time_offset;
        Some(end_sec)
    }
}

fn is_cjk(language: &str) -> bool {
    matches!(language, "ja" | "zh" | "yue" | "th" | "lo" | "my")
}
