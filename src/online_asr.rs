//! Streaming Whisper processor that drives an internal hypothesis buffer.
//!
//! Mirrors the `OnlineASRProcessor` in `ufal/whisper_streaming`: holds a
//! rolling 16 kHz f32 audio buffer, runs Whisper on the whole buffer
//! every `process` call, feeds word-level hypotheses into the
//! LocalAgreement-2 buffer, and trims old audio once committed words
//! pile up past `buffer_trimming_sec`.
//!
//! The processor keeps a shared handle to its `WhisperContext`, so
//! callers can load a model once with [`OnlineAsrModel`] and create
//! multiple independent processor states from it without seeing a
//! `whisper-rs` type.
//!
//! When constructed with VAD, the processor also runs Silero VAD on
//! every incoming chunk: silent samples are skipped, and after
//! `silence_reset_sec` of trailing silence the Whisper decoder state is
//! reset so the next utterance starts fresh. Committed timestamps stay
//! continuous across resets because the processor advances
//! `total_audio_sec` on every chunk (speech or silence) and uses it as
//! the new time offset.

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    WhisperVadContext, WhisperVadContextParams, WhisperVadParams,
};

use crate::error::Error;
use crate::hypothesis_buffer::{HypothesisBuffer, Word};

/// Audio sample rate expected by [`OnlineAsrProcessor::insert_audio_chunk`].
pub const SAMPLE_RATE: usize = 16_000;
const BUFFER_TRIMMING_SEC: f64 = 15.0;
const PROMPT_CHAR_BUDGET: usize = 200;

/// Whisper decoding strategy used for each streaming pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecodingStrategy {
    /// Greedy decoding. Lower CPU cost, usually less stable across
    /// overlapping streaming windows.
    Greedy {
        /// Number of candidates considered by whisper.cpp.
        best_of: i32,
    },
    /// Beam search. This is the default because LocalAgreement-2 relies
    /// on repeated passes producing a stable prefix.
    BeamSearch {
        /// Beam width. whisper.cpp clamps values below 1 to 1.
        beam_size: i32,
        /// Beam-search patience. `-1.0` matches whisper.cpp defaults.
        patience: f32,
    },
}

impl Default for DecodingStrategy {
    fn default() -> Self {
        Self::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        }
    }
}

impl DecodingStrategy {
    fn to_whisper(self) -> SamplingStrategy {
        match self {
            DecodingStrategy::Greedy { best_of } => SamplingStrategy::Greedy { best_of },
            DecodingStrategy::BeamSearch {
                beam_size,
                patience,
            } => SamplingStrategy::BeamSearch {
                beam_size,
                patience,
            },
        }
    }
}

/// Tunable parameters for a streaming ASR processor.
///
/// The input sample rate is intentionally not configurable: Whisper
/// expects 16 kHz audio, exposed as [`SAMPLE_RATE`]. Resample before
/// calling [`OnlineAsrProcessor::insert_audio_chunk`] if your source
/// uses another rate.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineAsrConfig {
    /// Whisper language code (`"en"`, `"ja"`, …) or `"auto"` for
    /// autodetect.
    pub language: String,
    /// Decoder strategy used for every `process` pass.
    pub decoding_strategy: DecodingStrategy,
    /// CPU threads used by Whisper inference.
    pub n_threads: i32,
    /// Rolling audio is trimmed after this many seconds when a stable
    /// cut point is available.
    pub buffer_trimming_sec: f64,
    /// Maximum number of prior committed characters passed as Whisper's
    /// initial prompt.
    pub prompt_char_budget: usize,
}

impl OnlineAsrConfig {
    /// Create a config using crate defaults for a language.
    pub fn new(language: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            decoding_strategy: DecodingStrategy::default(),
            n_threads: default_n_threads(),
            buffer_trimming_sec: BUFFER_TRIMMING_SEC,
            prompt_char_budget: PROMPT_CHAR_BUDGET,
        }
    }

    /// Set the decoder strategy.
    pub fn with_decoding_strategy(mut self, decoding_strategy: DecodingStrategy) -> Self {
        self.decoding_strategy = decoding_strategy;
        self
    }

    /// Set the Whisper inference thread count.
    pub fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads;
        self
    }

    /// Set the rolling buffer trimming threshold, in seconds.
    pub fn with_buffer_trimming_sec(mut self, buffer_trimming_sec: f64) -> Self {
        self.buffer_trimming_sec = buffer_trimming_sec;
        self
    }

    /// Set the initial-prompt character budget.
    pub fn with_prompt_char_budget(mut self, prompt_char_budget: usize) -> Self {
        self.prompt_char_budget = prompt_char_budget;
        self
    }
}

impl Default for OnlineAsrConfig {
    fn default() -> Self {
        Self::new("auto")
    }
}

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
            n_threads: default_n_threads(),
        }
    }
}

/// Result of one streaming processing pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProcessOutput {
    /// Words committed by this pass.
    pub committed: Vec<Word>,
    /// Current uncommitted hypothesis after this pass.
    pub tentative: Vec<Word>,
    /// True when `committed` came from a VAD utterance boundary flush
    /// instead of a normal LocalAgreement-2 prefix match.
    pub finalized_by_vad: bool,
}

impl ProcessOutput {
    /// Whether this pass produced no committed or tentative words.
    pub fn is_empty(&self) -> bool {
        self.committed.is_empty() && self.tentative.is_empty()
    }
}

/// Loaded Whisper model that can create multiple streaming processors
/// without reloading model weights from disk.
#[derive(Clone, Debug)]
pub struct OnlineAsrModel {
    ctx: Arc<WhisperContext>,
}

impl OnlineAsrModel {
    /// Load a Whisper model from disk.
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, Error> {
        let ctx = WhisperContext::new_with_params(
            model_path.as_ref(),
            WhisperContextParameters::default(),
        )
        .map_err(|e| Error::ModelLoad(e.to_string()))?;
        Ok(Self { ctx: Arc::new(ctx) })
    }

    /// Create a processor with default ASR settings for `language`.
    pub fn create_processor(&self, language: &str) -> Result<OnlineAsrProcessor, Error> {
        self.create_processor_with_config(OnlineAsrConfig::new(language))
    }

    /// Create a processor with explicit ASR settings.
    pub fn create_processor_with_config(
        &self,
        config: OnlineAsrConfig,
    ) -> Result<OnlineAsrProcessor, Error> {
        OnlineAsrProcessor::from_context(self.ctx.clone(), config, None)
    }

    /// Create a processor with default ASR settings and a shared VAD model.
    pub fn create_processor_with_vad(
        &self,
        language: &str,
        vad_model: &VadModel,
    ) -> Result<OnlineAsrProcessor, Error> {
        self.create_processor_with_config_and_vad(OnlineAsrConfig::new(language), vad_model)
    }

    /// Create a processor with explicit ASR settings and a shared VAD model.
    pub fn create_processor_with_config_and_vad(
        &self,
        config: OnlineAsrConfig,
        vad_model: &VadModel,
    ) -> Result<OnlineAsrProcessor, Error> {
        OnlineAsrProcessor::from_context(
            self.ctx.clone(),
            config,
            Some(VadState::new(vad_model.clone())),
        )
    }
}

/// Loaded Silero VAD model that can be shared by multiple processors.
///
/// Sharing avoids loading the VAD weights more than once. VAD passes are
/// serialized internally because `whisper.cpp` exposes VAD inference as a
/// mutable context operation.
#[derive(Clone, Debug)]
pub struct VadModel {
    ctx: Arc<Mutex<WhisperVadContext>>,
    config: VadConfig,
    silence_reset_samples: usize,
}

impl VadModel {
    /// Load a VAD model with [`VadConfig::default`].
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, Error> {
        Self::load_with_config(model_path, VadConfig::default())
    }

    /// Load a VAD model with explicit VAD settings.
    ///
    /// `config.n_threads` is applied while loading the VAD context; the
    /// remaining fields are used by processors that share this model.
    pub fn load_with_config<P: AsRef<Path>>(
        model_path: P,
        config: VadConfig,
    ) -> Result<Self, Error> {
        let vad_path = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| Error::VadModelLoad("VAD model path is not valid UTF-8".to_string()))?;
        let mut vad_params = WhisperVadContextParams::default();
        vad_params.set_n_threads(config.n_threads.max(1));
        let ctx = WhisperVadContext::new(vad_path, vad_params)
            .map_err(|e| Error::VadModelLoad(e.to_string()))?;
        let silence_reset_samples =
            (config.silence_reset_sec.max(0.0) * SAMPLE_RATE as f64) as usize;
        Ok(Self {
            ctx: Arc::new(Mutex::new(ctx)),
            config,
            silence_reset_samples,
        })
    }

    /// VAD settings used by processors that share this model.
    pub fn config(&self) -> VadConfig {
        self.config
    }

    fn segments_from_samples(
        &self,
        params: WhisperVadParams,
        pcm: &[f32],
    ) -> Result<whisper_rs::WhisperVadSegments, Error> {
        let mut ctx = self
            .ctx
            .lock()
            .map_err(|_| Error::Vad("VAD model lock was poisoned".to_string()))?;
        ctx.segments_from_samples(params, pcm)
            .map_err(|e| Error::Vad(e.to_string()))
    }
}

/// Per-processor VAD state: shared model plus the rolling silence
/// counter and the latched "reset on next iter" flag.
struct VadState {
    model: VadModel,
    silence_samples: usize,
    reset_pending: bool,
}

impl VadState {
    fn new(model: VadModel) -> Self {
        Self {
            model,
            silence_samples: 0,
            reset_pending: false,
        }
    }
}

/// Streaming wrapper around `whisper-rs` implementing LocalAgreement-2.
pub struct OnlineAsrProcessor {
    // Shared model context. The live WhisperState is independent per
    // processor; the model weights are reused through OnlineAsrModel.
    ctx: Arc<WhisperContext>,
    state: WhisperState,
    audio_buffer: Vec<f32>,
    buffer_time_offset: f64,
    transcript_buffer: HypothesisBuffer,
    committed: Vec<Word>,
    sep: &'static str,
    config: OnlineAsrConfig,

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
        OnlineAsrModel::load(model_path)?.create_processor(language)
    }

    /// Load a Whisper model from disk and create a streaming processor
    /// with explicit ASR settings.
    pub fn from_model_path_with_config<P: AsRef<Path>>(
        model_path: P,
        config: OnlineAsrConfig,
    ) -> Result<Self, Error> {
        OnlineAsrModel::load(model_path)?.create_processor_with_config(config)
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
        let model = OnlineAsrModel::load(model_path)?;
        let vad_model = VadModel::load_with_config(vad_model_path, vad_config)?;
        model.create_processor_with_vad(language, &vad_model)
    }

    /// Internal constructor shared by both public entry points. Builds
    /// the initial Whisper state and seeds all per-utterance counters.
    fn from_context(
        ctx: Arc<WhisperContext>,
        config: OnlineAsrConfig,
        vad: Option<VadState>,
    ) -> Result<Self, Error> {
        let state = ctx
            .create_state()
            .map_err(|e| Error::StateInit(e.to_string()))?;
        let sep = if is_cjk(&config.language) { "" } else { " " };
        Ok(Self {
            ctx,
            state,
            audio_buffer: Vec::new(),
            buffer_time_offset: 0.0,
            transcript_buffer: HypothesisBuffer::new(),
            committed: Vec::new(),
            sep,
            config,
            vad,
            total_audio_sec: 0.0,
            speaking: false,
        })
    }

    /// Append 16 kHz mono f32 PCM samples to the rolling buffer. With
    /// VAD enabled the chunk is first classified: silent chunks are
    /// dropped, and once trailing silence has accumulated past
    /// `silence_reset_sec` a reset is latched for the next
    /// [`Self::process`].
    pub fn insert_audio_chunk(&mut self, pcm: &[f32]) -> Result<(), Error> {
        let chunk_len = pcm.len();

        if let Some(vad) = self.vad.as_mut() {
            let config = vad.model.config();
            let mut params = WhisperVadParams::default();
            params.set_threshold(config.threshold);
            params.set_min_speech_duration(config.min_speech_ms);
            params.set_min_silence_duration(config.min_silence_ms);
            let segments = vad.model.segments_from_samples(params, pcm)?;
            let has_speech = segments.num_segments() > 0;

            if has_speech {
                self.audio_buffer.extend_from_slice(pcm);
                vad.silence_samples = 0;
                self.speaking = true;
            } else if self.speaking {
                vad.silence_samples += chunk_len;
                if vad.silence_samples >= vad.model.silence_reset_samples {
                    vad.reset_pending = true;
                }
            }
        } else {
            self.audio_buffer.extend_from_slice(pcm);
        }

        self.total_audio_sec += chunk_len as f64 / SAMPLE_RATE as f64;
        Ok(())
    }

    /// Run one Whisper pass over the rolling buffer.
    ///
    /// The returned [`ProcessOutput`] contains newly committed words,
    /// the current tentative hypothesis, and whether a VAD utterance
    /// boundary forced a final flush.
    pub fn process(&mut self) -> Result<ProcessOutput, Error> {
        // Handle a pending VAD reset before doing any new inference.
        // The flushed tentative words are returned as if they had been
        // committed — semantically this is what the example pipeline
        // already does at utterance boundaries.
        if self.vad.as_ref().is_some_and(|v| v.reset_pending) {
            let flushed = self.transcript_buffer.complete();
            self.reset_for_next_utterance();
            return Ok(ProcessOutput {
                committed: flushed,
                tentative: Vec::new(),
                finalized_by_vad: true,
            });
        }

        if self.audio_buffer.is_empty() {
            return Ok(ProcessOutput {
                committed: Vec::new(),
                tentative: self.tentative(),
                finalized_by_vad: false,
            });
        }

        let prompt = self.build_init_prompt();

        // Match the reference whisper_streaming defaults (BeamSearch with
        // beam_size=5). Greedy is cheaper but produces less stable
        // predictions across overlapping audio windows, which defeats
        // LocalAgreement-2 (it relies on the same prefix being predicted
        // twice in a row).
        let mut params = FullParams::new(self.config.decoding_strategy.to_whisper());
        if !self.config.language.is_empty() && self.config.language != "auto" {
            params.set_language(Some(&self.config.language));
        }
        params.set_n_threads(self.config.n_threads.max(1));
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

        Ok(ProcessOutput {
            committed,
            tentative: self.tentative(),
            finalized_by_vad: false,
        })
    }

    /// Run one Whisper pass over the rolling buffer and return only the
    /// words that were just committed by LocalAgreement-2.
    #[deprecated(
        since = "0.1.0",
        note = "use process() to get committed words, tentative words, and VAD finalization metadata"
    )]
    pub fn process_iter(&mut self) -> Result<Vec<Word>, Error> {
        self.process().map(|output| output.committed)
    }

    /// Consume the still-tentative buffer at end of stream.
    ///
    /// Calling this marks the current audio buffer as finished and clears
    /// the in-flight hypothesis. A second call returns no words unless more
    /// audio has been inserted.
    pub fn finish(&mut self) -> Vec<Word> {
        let remaining = self.transcript_buffer.drain_complete();
        self.buffer_time_offset += self.audio_buffer.len() as f64 / SAMPLE_RATE as f64;
        self.audio_buffer.clear();
        self.committed.extend(remaining.iter().cloned());
        self.transcript_buffer
            .reset_to_time(self.buffer_time_offset);
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
        // does, we keep the old state — the next process pass
        // will then operate on an empty audio buffer and return [].
        if let Ok(new_state) = self.ctx.create_state() {
            self.state = new_state;
        }
        self.audio_buffer.clear();
        self.transcript_buffer.reset_to_time(self.total_audio_sec);
        self.buffer_time_offset = self.total_audio_sec;
        self.committed.clear();
        self.speaking = false;
        if let Some(vad) = self.vad.as_mut() {
            vad.silence_samples = 0;
            vad.reset_pending = false;
        }
    }

    /// Reverse-walk committed words and join the most recent ones
    /// into an init-prompt string, respecting the configured character
    /// budget,
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
            if chars + bump > self.config.prompt_char_budget {
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
    /// the current audio buffer; the offset is added later by the
    /// hypothesis buffer.
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
    /// 1. End of a completed Whisper segment that is still within the
    ///    committed LocalAgreement prefix.
    /// 2. End of the last committed word.
    ///
    /// Both strategies are bounded by committed text. Trimming beyond the
    /// last committed word would discard audio that LocalAgreement has not
    /// confirmed yet, causing words to disappear from later agreement checks.
    fn maybe_trim(&mut self) {
        let buf_secs = self.audio_buffer.len() as f64 / SAMPLE_RATE as f64;
        if buf_secs <= self.config.buffer_trimming_sec {
            return;
        }

        let cut_time = self
            .segment_end_cut()
            .or_else(|| self.commit_based_cut())
            .filter(|&t| t > self.buffer_time_offset);
        let Some(cut_time) = cut_time else {
            return;
        };

        self.transcript_buffer.pop_committed(cut_time);
        let cut_samples = (((cut_time - self.buffer_time_offset) * SAMPLE_RATE as f64) as usize)
            .min(self.audio_buffer.len());
        self.audio_buffer.drain(..cut_samples);
        self.buffer_time_offset = cut_time;
        // Keep the start-time filter from rejecting future words that
        // legitimately start near the new buffer origin.
        self.transcript_buffer.advance_last_committed_time(cut_time);
    }

    fn commit_based_cut(&self) -> Option<f64> {
        self.committed
            .last()
            .map(|w| w.end)
            .filter(|&e| e > self.buffer_time_offset)
    }

    fn segment_end_cut(&self) -> Option<f64> {
        let last_committed_end = self.committed.last()?.end;
        let n = self.state.full_n_segments();
        if n < 2 {
            return None;
        }
        let mut segment_ends = Vec::with_capacity(n as usize);
        for seg_idx in 0..n {
            let seg = self.state.get_segment(seg_idx)?;
            segment_ends.push(seg.end_timestamp() as f64 / 100.0);
        }
        segment_cut_within_committed(&segment_ends, self.buffer_time_offset, last_committed_end)
    }
}

fn segment_cut_within_committed(
    segment_ends: &[f64],
    buffer_time_offset: f64,
    last_committed_end: f64,
) -> Option<f64> {
    if segment_ends.len() < 2 {
        return None;
    }

    let mut idx = segment_ends.len() - 2;
    while idx > 0 && segment_ends[idx] + buffer_time_offset > last_committed_end {
        idx -= 1;
    }

    let cut_time = segment_ends[idx] + buffer_time_offset;
    (cut_time > buffer_time_offset && cut_time <= last_committed_end).then_some(cut_time)
}

fn is_cjk(language: &str) -> bool {
    matches!(language, "ja" | "zh" | "yue" | "th" | "lo" | "my")
}

fn default_n_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_cut_never_exceeds_last_committed_word() {
        let cut = segment_cut_within_committed(&[4.0, 8.0, 12.0], 0.0, 6.0);

        assert_eq!(cut, Some(4.0));
    }

    #[test]
    fn segment_cut_uses_second_to_last_when_committed() {
        let cut = segment_cut_within_committed(&[4.0, 8.0, 12.0], 0.0, 9.0);

        assert_eq!(cut, Some(8.0));
    }

    #[test]
    fn segment_cut_returns_none_without_committed_audio_after_offset() {
        let cut = segment_cut_within_committed(&[4.0, 8.0, 12.0], 5.0, 8.0);

        assert_eq!(cut, None);
    }
}
