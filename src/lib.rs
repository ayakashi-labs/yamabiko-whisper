//! Streaming speech recognition on top of `whisper-rs`, using the
//! LocalAgreement-2 policy from Macháček et al. 2023.
//!
//! `whisper-rs` is fully encapsulated: callers do not need to add it
//! to their own `Cargo.toml` or import any of its types. Build a
//! processor with [`OnlineAsrProcessor::from_model_path`], feed it
//! 16 kHz mono f32 PCM via [`OnlineAsrProcessor::insert_audio_chunk`],
//! and pull committed words out with
//! [`OnlineAsrProcessor::process_iter`].

mod error;
mod hypothesis_buffer;
mod online_asr;

pub use error::Error;
pub use hypothesis_buffer::Word;
pub use online_asr::{OnlineAsrProcessor, VadConfig};

/// Forward whisper.cpp / GGML / VAD logs to a `log` / `tracing`
/// backend. Without calling this they are silently dropped, which is
/// usually what you want; call this once at startup if you do want to
/// see them. Thin wrapper around `whisper_rs::install_logging_hooks`
/// so callers do not need to depend on `whisper-rs` directly.
pub fn install_log_hooks() {
    whisper_rs::install_logging_hooks();
}
