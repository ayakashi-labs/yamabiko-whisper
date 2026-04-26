//! Batch transcription of a 16 kHz mono WAV via the streaming API.
//!
//! This example feeds the entire file to `OnlineAsrProcessor` in one
//! shot and then drains both the LocalAgreement-confirmed and the
//! still-tentative words. It serves as a quick sanity check that does
//! not need a microphone.
//!
//! Usage:
//!     cargo run --example simple_transcribe -- <model.bin> <audio.wav> [language=auto]

use anyhow::{Context, Result, anyhow};
use hound::WavReader;
use local_agreement_whisper::{OnlineAsrProcessor, Word};

fn main() -> Result<()> {
    let model_path = std::env::args().nth(1).ok_or_else(|| {
        anyhow!("usage: simple_transcribe <model.bin> <audio.wav> [language=auto]")
    })?;
    let audio_path = std::env::args()
        .nth(2)
        .ok_or_else(|| anyhow!("missing audio path"))?;
    let language = std::env::args().nth(3).unwrap_or_else(|| "auto".into());

    // Decode 16-bit PCM WAV (assumed 16 kHz mono) to f32 in [-1, 1].
    let audio_data: Vec<f32> = WavReader::open(&audio_path)
        .with_context(|| format!("failed to open wav: {audio_path}"))?
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let mut processor = OnlineAsrProcessor::from_model_path(&model_path, &language)?;
    processor.insert_audio_chunk(&audio_data)?;

    // Two passes so LocalAgreement-2 can confirm a prefix; one pass
    // alone never commits anything because the policy needs the same
    // prefix to be predicted twice in a row.
    let _ = processor.process_iter()?;
    let committed = processor.process_iter()?;
    let final_words = processor.finish();

    for w in committed.iter().chain(final_words.iter()) {
        print_word(w);
    }
    Ok(())
}

/// Print one word in the same `[start - end]: text` centisecond format
/// the streaming example uses.
fn print_word(w: &Word) {
    println!(
        "[{} - {}]: {}",
        (w.start * 100.0).round() as i64,
        (w.end * 100.0).round() as i64,
        w.text
    );
}
