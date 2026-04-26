//! Real-time microphone transcription with LocalAgreement-2 streaming.
//!
//! Pipeline:
//!     cpal default input device
//!         -> per-callback mono mix + sample-format conversion to f32
//!         -> std::sync::mpsc to the main thread
//!         -> on-line linear-interpolation resampler to 16 kHz
//!         -> optional Silero VAD gate (whisper.cpp's built-in)
//!         -> OnlineAsrProcessor (~1.0 s of new audio per pass)
//!         -> committed words printed as `[<start> - <end>]: <text>` in
//!            centiseconds, matching the simple_transcribe example
//!            (stdout)
//!         -> tentative hypothesis rendered with `\r` overlay (stderr)
//!
//! Usage:
//!     cargo run --release --example streaming_mic -- \
//!         <model.bin> [language=auto] [vad-model.bin]
//!
//! When a VAD model path is provided, silent chunks are dropped and the
//! Whisper processor is reset after a short trailing silence so the next
//! utterance starts with a fresh `[0.00]` timeline. Without a VAD model
//! the processor consumes every chunk (which is what the paper describes
//! but tends to hallucinate on long silences when running off a live
//! microphone).
//!
//! Press Ctrl-C to stop. The processor's tentative buffer is then flushed
//! once and printed as a final committed line.

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat};
use whisper_rs::{
    WhisperContext, WhisperContextParameters, WhisperVadContext, WhisperVadContextParams,
    WhisperVadParams, install_logging_hooks,
};

use local_agreement_whisper::{OnlineAsrProcessor, Word};

/// Run a Whisper pass once we have at least this many seconds of new audio.
const MIN_CHUNK_SEC: f64 = 1.0;
/// Target sampling rate for whisper.cpp (fixed by the model).
const TARGET_SR: u32 = 16_000;
/// Trailing silence (in seconds) that ends an utterance and resets the
/// streaming processor. Only used when a VAD model is supplied.
const SILENCE_RESET_SEC: f64 = 2.0;

fn main() -> Result<()> {
    // Route whisper.cpp / GGML / VAD logs through whisper-rs's logging
    // hooks. Without a `log` or `tracing` backend wired up they are
    // silently dropped, which is what we want here — those logs would
    // otherwise spam stderr several times per second.
    install_logging_hooks();

    let mut args = std::env::args().skip(1);
    let model_path = args.next().ok_or_else(|| {
        anyhow!("usage: streaming_mic <model.bin> [language=auto] [vad-model.bin]")
    })?;
    let language = args.next().unwrap_or_else(|| "auto".to_string());
    let vad_model_path = args.next();

    eprintln!("loading model: {model_path}");
    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .map_err(|e| anyhow!("failed to load whisper model: {e}"))?;
    let mut processor = OnlineAsrProcessor::new(&ctx, &language);

    let mut vad = match vad_model_path.as_deref() {
        Some(path) => {
            eprintln!("loading VAD model: {path}");
            let mut vad_params = WhisperVadContextParams::default();
            vad_params.set_n_threads(
                std::thread::available_parallelism()
                    .map(|n| n.get() as i32)
                    .unwrap_or(4),
            );
            let ctx = WhisperVadContext::new(path, vad_params)
                .map_err(|e| anyhow!("failed to load VAD model: {e}"))?;
            Some(ctx)
        }
        None => {
            eprintln!("no VAD model supplied; passing every chunk through Whisper");
            None
        }
    };

    // Pick the default microphone and its native config.
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device"))?;
    let supported_config = device
        .default_input_config()
        .context("failed to query default input config")?;
    let input_sr = supported_config.sample_rate();
    let channels = supported_config.channels() as usize;
    let sample_format = supported_config.sample_format();
    let stream_config: cpal::StreamConfig = supported_config.into();

    eprintln!(
        "input: {} Hz, {} ch, {:?} -> resampling to {} Hz mono",
        input_sr, channels, sample_format, TARGET_SR
    );

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let err_fn = |err| eprintln!("stream error: {err}");

    // Build a typed input stream per cpal sample format. Each callback
    // converts to f32, downmixes to mono, and forwards a Vec<f32> chunk.
    let stream = match sample_format {
        SampleFormat::F32 => {
            let tx = tx.clone();
            device.build_input_stream::<f32, _, _>(
                &stream_config,
                move |data, _| {
                    let _ = tx.send(downmix(data, channels));
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let tx = tx.clone();
            device.build_input_stream::<i16, _, _>(
                &stream_config,
                move |data, _| {
                    let _ = tx.send(downmix(data, channels));
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I32 => {
            let tx = tx.clone();
            device.build_input_stream::<i32, _, _>(
                &stream_config,
                move |data, _| {
                    let _ = tx.send(downmix(data, channels));
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let tx = tx.clone();
            device.build_input_stream::<u16, _, _>(
                &stream_config,
                move |data, _| {
                    let _ = tx.send(downmix(data, channels));
                },
                err_fn,
                None,
            )?
        }
        other => bail!("unsupported sample format: {:?}", other),
    };
    drop(tx);

    stream.play().context("failed to start input stream")?;

    let running = Arc::new(AtomicBool::new(true));
    {
        let running = running.clone();
        ctrlc::set_handler(move || running.store(false, Ordering::SeqCst))
            .context("failed to install Ctrl-C handler")?;
    }

    eprintln!("speak now (Ctrl-C to stop)…");

    let mut resampler = LinearResampler::new(input_sr, TARGET_SR);
    let mut pending: Vec<f32> = Vec::new();
    let min_samples = (MIN_CHUNK_SEC * TARGET_SR as f64) as usize;
    let mut tentative_visible = false;
    // Number of consecutive silent samples at 16 kHz; only meaningful when
    // VAD is on. Used to decide when an utterance has ended so we can wipe
    // the processor state before the next one begins.
    let mut silence_samples: usize = 0;
    // Whether the processor currently holds any active speech. Drives the
    // utterance-end reset and the final-flush behaviour at Ctrl-C.
    let mut speaking = false;
    // Total seconds of audio observed since program start. Every chunk
    // (speech or silence) advances this counter, and it is used as the
    // offset for any rebuilt processor so committed timestamps remain
    // continuous across utterance resets instead of restarting at 0.
    let mut total_audio_sec: f64 = 0.0;
    let silence_reset_samples = (SILENCE_RESET_SEC * TARGET_SR as f64) as usize;

    while running.load(Ordering::SeqCst) {
        // Drain everything currently queued, with a short wait so the loop
        // does not spin when the mic is silent.
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => pending.extend(resampler.process(&chunk)),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        while let Ok(chunk) = rx.try_recv() {
            pending.extend(resampler.process(&chunk));
        }

        if pending.len() >= min_samples {
            let chunk_len = pending.len();
            let has_speech = match vad.as_mut() {
                Some(vad_ctx) => chunk_contains_speech(vad_ctx, &pending)?,
                None => true,
            };

            if has_speech {
                processor.insert_audio_chunk(&pending);
                let committed = processor.process_iter();
                render(
                    &committed,
                    processor.tentative(),
                    processor.sep(),
                    &mut tentative_visible,
                );
                silence_samples = 0;
                speaking = true;
            } else if speaking {
                silence_samples += chunk_len;
                if silence_samples >= silence_reset_samples {
                    // End-of-utterance: flush any tentative words, then
                    // build a fresh processor seeded with the current
                    // wall-clock offset so the next utterance picks up
                    // where this one left off on the timeline.
                    let final_words = processor.finish();
                    if tentative_visible {
                        eprint!("\r\x1b[K");
                        tentative_visible = false;
                    }
                    if let (Some(first), Some(last)) = (final_words.first(), final_words.last()) {
                        let joined = join_words(&final_words, processor.sep());
                        println!("[{} - {}]: {}", to_cs(first.start), to_cs(last.end), joined);
                        let _ = std::io::stdout().flush();
                    }
                    let next_offset = total_audio_sec + chunk_len as f64 / TARGET_SR as f64;
                    processor = OnlineAsrProcessor::with_offset(&ctx, &language, next_offset);
                    silence_samples = 0;
                    speaking = false;
                }
            }
            total_audio_sec += chunk_len as f64 / TARGET_SR as f64;
            pending.clear();
        }
    }

    // Stop capture and flush whatever remains.
    drop(stream);
    if !pending.is_empty() && speaking {
        processor.insert_audio_chunk(&pending);
        let committed = processor.process_iter();
        render(
            &committed,
            processor.tentative(),
            processor.sep(),
            &mut tentative_visible,
        );
    }
    let final_words = processor.finish();
    if tentative_visible {
        // Wipe the in-flight overlay before the final committed line.
        eprint!("\r\x1b[K");
    }
    if let (Some(first), Some(last)) = (final_words.first(), final_words.last()) {
        let joined = join_words(&final_words, processor.sep());
        println!("[{} - {}]: {}", to_cs(first.start), to_cs(last.end), joined);
    }

    Ok(())
}

/// Convert a seconds-valued timestamp to centiseconds (10 ms units), the
/// same unit `simple_transcribe.rs` prints.
fn to_cs(sec: f64) -> i64 {
    (sec * 100.0).round() as i64
}

/// Run the Silero VAD over a 16 kHz mono f32 chunk and return true if any
/// speech segment was detected. Uses default `WhisperVadParams` which
/// match whisper.cpp's recommended settings (250 ms min speech, 100 ms min
/// silence, 0.5 probability threshold).
fn chunk_contains_speech(vad: &mut WhisperVadContext, samples: &[f32]) -> Result<bool> {
    let segments = vad
        .segments_from_samples(WhisperVadParams::default(), samples)
        .map_err(|e| anyhow!("VAD failed: {e}"))?;
    Ok(segments.num_segments() > 0)
}

/// Convert any cpal sample type to f32 and average all channels into one
/// mono frame. Channel count of 0 should not happen in practice.
fn downmix<T>(data: &[T], channels: usize) -> Vec<f32>
where
    T: cpal::SizedSample,
    f32: FromSample<T>,
{
    if channels <= 1 {
        return data.iter().map(|&s| f32::from_sample(s)).collect();
    }
    data.chunks_exact(channels)
        .map(|frame| {
            let sum: f32 = frame.iter().map(|&s| f32::from_sample(s)).sum();
            sum / channels as f32
        })
        .collect()
}

fn join_words(words: &[Word], sep: &str) -> String {
    let parts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    parts.join(sep)
}

/// Print just-committed words as a new line; redraw the tentative overlay
/// underneath so the user always sees both: a stable history above and a
/// moving prediction below. Committed lines use the same
/// `[<start> - <end>]: <text>` centisecond format as
/// `simple_transcribe.rs`, taking the start of the first and end of the
/// last word in the batch.
fn render(committed: &[Word], tentative: Vec<Word>, sep: &str, tentative_visible: &mut bool) {
    if let (Some(first), Some(last)) = (committed.first(), committed.last()) {
        // Erase the tentative line first so the new committed line lands
        // on its own row instead of being prepended to the overlay text.
        if *tentative_visible {
            eprint!("\r\x1b[K");
            *tentative_visible = false;
        }
        let joined = join_words(committed, sep);
        println!("[{} - {}]: {}", to_cs(first.start), to_cs(last.end), joined);
        let _ = std::io::stdout().flush();
    }

    // Re-draw tentative hypothesis on stderr so it stays visually distinct
    // from the committed stdout history.
    if !tentative.is_empty() {
        let joined = join_words(&tentative, sep);
        eprint!("\r  …(tentative): {joined}\x1b[K");
        let _ = std::io::stderr().flush();
        *tentative_visible = true;
    } else if *tentative_visible {
        eprint!("\r\x1b[K");
        let _ = std::io::stderr().flush();
        *tentative_visible = false;
    }
}

/// Tiny streaming linear-interpolation resampler. Quality is more than
/// adequate for ASR: speech energy lives below 4 kHz and Whisper expects
/// a 16 kHz log-mel spectrogram, so anti-aliasing is taken care of by
/// the model's own front-end. Keeps one sample of state across calls so
/// the interpolation is continuous at chunk boundaries.
struct LinearResampler {
    ratio: f64,
    pos: f64,
    last: f32,
}

impl LinearResampler {
    fn new(input_sr: u32, output_sr: u32) -> Self {
        Self {
            ratio: input_sr as f64 / output_sr as f64,
            pos: 0.0,
            last: 0.0,
        }
    }

    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        // If sample rates already match, fast-path the chunk verbatim.
        if (self.ratio - 1.0).abs() < 1e-9 {
            self.last = *input.last().unwrap();
            return input.to_vec();
        }

        let mut out = Vec::with_capacity(((input.len() as f64) / self.ratio) as usize + 1);
        let len = input.len() as f64;
        while self.pos < len {
            let i = self.pos.floor() as isize;
            let frac = (self.pos - self.pos.floor()) as f32;
            let next_idx = (i + 1) as usize;
            if next_idx >= input.len() {
                // Need a sample from the *next* chunk to interpolate; defer.
                break;
            }
            let a = if i < 0 { self.last } else { input[i as usize] };
            let b = input[next_idx];
            out.push(a * (1.0 - frac) + b * frac);
            self.pos += self.ratio;
        }
        // Re-base position so it stays in [-1, ratio) for the next call.
        self.pos -= input.len() as f64;
        self.last = *input.last().unwrap();
        out
    }
}
