//! Streaming Whisper processor that drives a [`HypothesisBuffer`].
//!
//! Mirrors the `OnlineASRProcessor` in `ufal/whisper_streaming`: holds a
//! rolling 16 kHz f32 audio buffer, runs Whisper on the whole buffer
//! every `process_iter` call, feeds word-level hypotheses into the
//! LocalAgreement-2 buffer, and trims old audio once committed words
//! pile up past `buffer_trimming_sec`.
//!
//! The processor owns its `WhisperContext` (whose internal state is
//! itself an `Arc<WhisperInnerContext>` shared with the live
//! `WhisperState`), so callers never see a `whisper-rs` type and the
//! processor has a `'static` lifetime.
//!
//! When constructed via [`OnlineAsrProcessor::from_model_path_with_vad`]
//! the processor also runs Silero VAD on every incoming chunk: silent
//! samples are skipped, and after `silence_reset_sec` of trailing
//! silence the Whisper decoder state is reset so the next utterance
//! starts fresh. Committed timestamps stay continuous across resets
//! because the processor advances `total_audio_sec` on every chunk
//! (speech or silence) and uses it as the new time offset.

use std::path::Path;

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    WhisperVadContext, WhisperVadContextParams, WhisperVadParams,
};

use crate::error::Error;
use crate::hypothesis_buffer::{HypothesisBuffer, Word};

const SAMPLING_RATE: usize = 16_000;
const BUFFER_TRIMMING_SEC: f64 = 15.0;
const PROMPT_CHAR_BUDGET: usize = 200;

/// Tunable parameters for the integrated Silero VAD. `Default` matches
/// whisper.cpp's recommended Silero settings (250 ms minimum speech,
/// 100 ms minimum silence, 0.5 probability threshold) plus a 2 s
/// silence-to-reset window suitable for live microphone use.
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Probability threshold to count a frame as speech (0.0–1.0).
    pub threshold: f32,
    /// Minimum speech segment duration, in milliseconds.
    pub min_speech_ms: i32,
    /// Minimum silence duration to end a segment, in milliseconds.
    pub min_silence_ms: i32,
    /// Trailing silence in seconds after which the processor's Whisper
    /// state is reset so the next utterance starts with a clean slate.
    pub silence_reset_sec: f64,
    /// CPU threads dedicated to the VAD model.
    pub n_threads: i32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 100,
            silence_reset_sec: 2.0,
            n_threads: std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4),
        }
    }
}

/// Per-processor VAD state: model context plus the rolling silence
/// counter and the latched "reset on next iter" flag.
struct VadState {
    ctx: WhisperVadContext,
    config: VadConfig,
    silence_samples: usize,
    silence_reset_samples: usize,
    reset_pending: bool,
}

/// Streaming wrapper around `whisper-rs` implementing LocalAgreement-2.
pub struct OnlineAsrProcessor {
    // Owned context. WhisperContext is internally Arc-shared so the
    // live WhisperState keeps it alive even if we ever drop this field.
    ctx: WhisperContext,
    state: WhisperState,
    audio_buffer: Vec<f32>,
    buffer_time_offset: f64,
    transcript_buffer: HypothesisBuffer,
    committed: Vec<Word>,
    sep: &'static str,
    language: String,
    n_threads: i32,
    buffer_trimming_sec: f64,

    vad: Option<VadState>,
    /// Total seconds of 16 kHz audio the processor has observed since
    /// construction (speech or silence). Used as the timeline anchor
    /// when the VAD reset rebuilds the decoder state, so committed
    /// timestamps stay continuous instead of restarting at 0.
    total_audio_sec: f64,
    /// True once the processor has seen any speech in the current
    /// utterance — drives the silence-reset condition (we never reset
    /// while the user has not started speaking yet).
    speaking: bool,
}

impl OnlineAsrProcessor {
    /// Load a Whisper model from disk and create a streaming processor.
    /// `language` is the Whisper language code (`"en"`, `"ja"`, …) or
    /// `"auto"` for autodetect. The model is loaded with default
    /// `WhisperContextParameters`; GPU acceleration is controlled at
    /// crate-feature build time.
    pub fn from_model_path<P: AsRef<Path>>(model_path: P, language: &str) -> Result<Self, Error> {
        let ctx = WhisperContext::new_with_params(
            model_path.as_ref(),
            WhisperContextParameters::default(),
        )
        .map_err(|e| Error::ModelLoad(e.to_string()))?;
        Self::from_context(ctx, language, None)
    }

    /// Same as [`Self::from_model_path`] but additionally loads a Silero
    /// VAD model. With VAD enabled, silent chunks are skipped and the
    /// internal Whisper state is automatically reset after
    /// `vad_config.silence_reset_sec` seconds of trailing silence.
    pub fn from_model_path_with_vad<P: AsRef<Path>, Q: AsRef<Path>>(
        model_path: P,
        language: &str,
        vad_model_path: Q,
        vad_config: VadConfig,
    ) -> Result<Self, Error> {
        let ctx = WhisperContext::new_with_params(
            model_path.as_ref(),
            WhisperContextParameters::default(),
        )
        .map_err(|e| Error::ModelLoad(e.to_string()))?;

        // WhisperVadContext::new takes &str (not AsRef<Path>); reject
        // paths whose UTF-8 form is lossy so the user gets a clear
        // error instead of a silently mangled filename.
        let vad_path = vad_model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| Error::VadModelLoad("VAD model path is not valid UTF-8".to_string()))?;
        let mut vad_params = WhisperVadContextParams::default();
        vad_params.set_n_threads(vad_config.n_threads);
        let vad_ctx = WhisperVadContext::new(vad_path, vad_params)
            .map_err(|e| Error::VadModelLoad(e.to_string()))?;

        let silence_reset_samples = (vad_config.silence_reset_sec * SAMPLING_RATE as f64) as usize;
        Self::from_context(
            ctx,
            language,
            Some(VadState {
                ctx: vad_ctx,
                config: vad_config,
                silence_samples: 0,
                silence_reset_samples,
                reset_pending: false,
            }),
        )
    }

    /// Internal constructor shared by both public entry points. Builds
    /// the initial Whisper state and seeds all per-utterance counters.
    fn from_context(
        ctx: WhisperContext,
        language: &str,
        vad: Option<VadState>,
    ) -> Result<Self, Error> {
        let state = ctx
            .create_state()
            .map_err(|e| Error::StateInit(e.to_string()))?;
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        let sep = if is_cjk(language) { "" } else { " " };
        Ok(Self {
            ctx,
            state,
            audio_buffer: Vec::new(),
            buffer_time_offset: 0.0,
            transcript_buffer: HypothesisBuffer::new(),
            committed: Vec::new(),
            sep,
            language: language.to_string(),
            n_threads,
            buffer_trimming_sec: BUFFER_TRIMMING_SEC,
            vad,
            total_audio_sec: 0.0,
            speaking: false,
        })
    }

    /// Append 16 kHz mono f32 PCM samples to the rolling buffer. With
    /// VAD enabled the chunk is first classified: silent chunks are
    /// dropped, and once trailing silence has accumulated past
    /// `silence_reset_sec` a reset is latched for the next
    /// [`process_iter`].
    pub fn insert_audio_chunk(&mut self, pcm: &[f32]) -> Result<(), Error> {
        let chunk_len = pcm.len();

        if let Some(vad) = self.vad.as_mut() {
            let mut params = WhisperVadParams::default();
            params.set_threshold(vad.config.threshold);
            params.set_min_speech_duration(vad.config.min_speech_ms);
            params.set_min_silence_duration(vad.config.min_silence_ms);
            let segments = vad
                .ctx
                .segments_from_samples(params, pcm)
                .map_err(|e| Error::Vad(e.to_string()))?;
            let has_speech = segments.num_segments() > 0;

            if has_speech {
                self.audio_buffer.extend_from_slice(pcm);
                vad.silence_samples = 0;
                self.speaking = true;
            } else if self.speaking {
                vad.silence_samples += chunk_len;
                if vad.silence_samples >= vad.silence_reset_samples {
                    vad.reset_pending = true;
                }
            }
        } else {
            self.audio_buffer.extend_from_slice(pcm);
        }

        self.total_audio_sec += chunk_len as f64 / SAMPLING_RATE as f64;
        Ok(())
    }

    /// Run one Whisper pass over the rolling buffer and return the
    /// words that were just committed by LocalAgreement-2. With VAD
    /// enabled, if a silence-reset is pending the still-tentative
    /// words are flushed first (and included in the returned vec)
    /// before the decoder is re-armed for the next utterance.
    pub fn process_iter(&mut self) -> Result<Vec<Word>, Error> {
        // Handle a pending VAD reset before doing any new inference.
        // The flushed tentative words are returned as if they had been
        // committed — semantically this is what the example pipeline
        // already does at utterance boundaries.
        if self.vad.as_ref().is_some_and(|v| v.reset_pending) {
            let flushed = self.transcript_buffer.complete();
            self.reset_for_next_utterance();
            return Ok(flushed);
        }

        if self.audio_buffer.is_empty() {
            return Ok(Vec::new());
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

        self.state
            .full(params, &self.audio_buffer)
            .map_err(|e| Error::Inference(e.to_string()))?;

        let words = self.extract_words();
        self.transcript_buffer
            .insert(words, self.buffer_time_offset);
        let committed = self.transcript_buffer.flush();
        self.committed.extend(committed.iter().cloned());

        self.maybe_trim();

        Ok(committed)
    }

    /// Drain the still-tentative buffer at end of stream.
    pub fn finish(&mut self) -> Vec<Word> {
        let remaining = self.transcript_buffer.complete();
        self.buffer_time_offset += self.audio_buffer.len() as f64 / SAMPLING_RATE as f64;
        remaining
    }

    /// Snapshot of the words that are predicted but not yet committed.
    /// Used by callers to render an in-flight overlay line.
    pub fn tentative(&self) -> Vec<Word> {
        self.transcript_buffer.complete()
    }

    /// Word separator for this language: `" "` for space-segmented
    /// languages, `""` for CJK.
    pub fn sep(&self) -> &'static str {
        self.sep
    }

    /// Rebuild the Whisper decoder state and clear all per-utterance
    /// buffers, anchoring the new timeline at `total_audio_sec` so
    /// subsequent committed words keep growing along the same axis.
    fn reset_for_next_utterance(&mut self) {
        // create_state should not fail on a healthy context; if it ever
        // does, we keep the old state — the next process_iter pass
        // will then operate on an empty audio buffer and return [].
        if let Ok(new_state) = self.ctx.create_state() {
            self.state = new_state;
        }
        self.audio_buffer.clear();
        self.transcript_buffer = HypothesisBuffer::new();
        self.transcript_buffer.last_committed_time = self.total_audio_sec;
        self.buffer_time_offset = self.total_audio_sec;
        self.committed.clear();
        self.speaking = false;
        if let Some(vad) = self.vad.as_mut() {
            vad.silence_samples = 0;
            vad.reset_pending = false;
        }
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
    /// `split_on_word(true)` having forced each segment to be exactly
    /// one word with proper t0/t1. Timestamps come back relative to
    /// the current audio buffer; the offset is added later inside
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
