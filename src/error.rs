//! Public error type for the crate.
//!
//! Wraps every failure path through `whisper-rs` as a plain string so
//! that callers never need to depend on `whisper_rs::WhisperError`
//! directly. Keeping the upstream type out of our public surface is
//! what lets users drop the `whisper-rs` dependency from their own
//! `Cargo.toml`.

use std::fmt;

/// All failures surfaced by `local_agreement_whisper`.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Loading the Whisper model file failed (path missing, bad format,
    /// GGML init failure, etc.).
    ModelLoad(String),
    /// Loading the Silero VAD model file failed.
    VadModelLoad(String),
    /// Allocating a Whisper decoder state failed.
    StateInit(String),
    /// Running Whisper inference on the current audio buffer failed.
    Inference(String),
    /// Running the VAD on the current audio chunk failed.
    Vad(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ModelLoad(msg) => write!(f, "failed to load whisper model: {msg}"),
            Error::VadModelLoad(msg) => write!(f, "failed to load VAD model: {msg}"),
            Error::StateInit(msg) => write!(f, "failed to create whisper state: {msg}"),
            Error::Inference(msg) => write!(f, "whisper inference failed: {msg}"),
            Error::Vad(msg) => write!(f, "VAD failed: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
