//! Streaming speech recognition on top of `whisper-rs`, using the
//! LocalAgreement-2 policy from Macháček et al. 2023.

pub mod hypothesis_buffer;
pub mod online_asr;

pub use hypothesis_buffer::{HypothesisBuffer, Word};
pub use online_asr::OnlineAsrProcessor;
