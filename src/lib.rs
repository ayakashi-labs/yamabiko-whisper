//! Streaming speech recognition on top of `whisper-rs`, using the
//! LocalAgreement-2 policy from Macháček et al. 2023.
//!
//! `whisper-rs` is fully encapsulated: callers do not need to add it
//! to their own `Cargo.toml` or import any of its types. Build a
//! processor with [`OnlineAsrModel::create_processor`] or
//! [`OnlineAsrProcessor::from_model_path`], feed it
//! 16 kHz mono f32 PCM via [`OnlineAsrProcessor::insert_audio_chunk`],
//! or let [`AsrPipeline`] handle downmixing, resampling, and chunking
//! for a microphone/file/network source.
//!
//! Accelerated whisper.cpp backends are exposed as Cargo features
//! (`cuda`, `vulkan`, `metal`, `coreml`, `hipblas`, `intel-sycl`,
//! `openblas`, `openmp`). Use [`BackendConfig`] when loading a model
//! to select a GPU device or force CPU execution.

mod audio;
mod error;
mod hypothesis_buffer;
mod online_asr;

pub use audio::{AsrPipeline, AudioInputConfig, AudioSample, LinearResampler, downmix_interleaved};
pub use error::Error;
pub use hypothesis_buffer::Word;
pub use online_asr::{
    BackendConfig, DecodingStrategy, OnlineAsrConfig, OnlineAsrModel, OnlineAsrProcessor,
    ProcessOutput, SAMPLE_RATE, VadConfig, VadModel,
};

/// Forward whisper.cpp / GGML / VAD logs to a `log` / `tracing`
/// backend. Without calling this they are silently dropped, which is
/// usually what you want; call this once at startup if you do want to
/// see them. Thin wrapper around `whisper_rs::install_logging_hooks`
/// so callers do not need to depend on `whisper-rs` directly.
pub fn install_log_hooks() {
    whisper_rs::install_logging_hooks();
}
