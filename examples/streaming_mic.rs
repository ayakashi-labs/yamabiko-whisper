//! Real-time microphone transcription with LocalAgreement-2 streaming.
//!
//! Pipeline:
//!     cpal default input device
//!         -> per-callback mono mix + sample-format conversion to f32
//!         -> std::sync::mpsc to the main thread
//!         -> on-line linear-interpolation resampler to 16 kHz
//!         -> OnlineAsrProcessor (~1.0 s of new audio per pass)
//!         -> committed words printed as `[<start> - <end>]: <text>`
//!            in centiseconds (stdout)
//!         -> tentative hypothesis rendered with `\r` overlay (stderr)
//!
//! Usage:
//!     cargo run --release --example streaming_mic -- \
//!         <model.bin> [language=auto] [vad-model.bin]
//!
//! The optional VAD model path enables Silero VAD inside the
//! processor: silent chunks are dropped and the Whisper state is
//! reset after a short trailing silence so the next utterance starts
//! with a fresh decoder timeline. Without it every chunk is forwarded
//! to Whisper directly.
//!
//! Press Ctrl-C to stop. The processor's tentative buffer is then
//! flushed once and printed as a final committed line.
//!
//! Note: this example never imports `whisper_rs` — everything that
//! used to require it (model loading, VAD context, log hooks) is now
//! exposed through `yamabiko_whisper`.

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat};

use yamabiko_whisper::{OnlineAsrModel, SAMPLE_RATE, VadConfig, VadModel, Word, install_log_hooks};

/// Run a Whisper pass once we have at least this many seconds of new audio.
const MIN_CHUNK_SEC: f64 = 1.0;
/// Target sampling rate for whisper.cpp (fixed by the model).
const TARGET_SR: u32 = SAMPLE_RATE as u32;

fn main() -> Result<()> {
    // Route whisper.cpp / GGML / VAD logs through a `log` / `tracing`
    // backend. With no backend wired up they get dropped, which is
    // what we want here — those logs would otherwise spam stderr
    // several times per second.
    install_log_hooks();

    let mut args = std::env::args().skip(1);
    let model_path = args.next().ok_or_else(|| {
        anyhow!("usage: streaming_mic <model.bin> [language=auto] [vad-model.bin]")
    })?;
    let language = args.next().unwrap_or_else(|| "auto".to_string());
    let vad_model_path = args.next();

    eprintln!("loading model: {model_path}");
    let model = OnlineAsrModel::load(&model_path)?;
    let mut processor = match vad_model_path.as_deref() {
        Some(vad_path) => {
            eprintln!("loading VAD model: {vad_path}");
            let vad_model = VadModel::load_with_config(vad_path, VadConfig::default())?;
            model.create_processor_with_vad(&language, &vad_model)?
        }
        None => {
            eprintln!("no VAD model supplied; passing every chunk through Whisper");
            model.create_processor(&language)?
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

    while running.load(Ordering::SeqCst) {
        // Drain everything currently queued, with a short wait so the
        // loop does not spin when the mic is silent.
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => pending.extend(resampler.process(&chunk)),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        while let Ok(chunk) = rx.try_recv() {
            pending.extend(resampler.process(&chunk));
        }

        if pending.len() >= min_samples {
            processor.insert_audio_chunk(&pending)?;
            let output = processor.process()?;
            render(
                &output.committed,
                output.tentative,
                processor.sep(),
                &mut tentative_visible,
            );
            pending.clear();
        }
    }

    // Stop capture and flush whatever remains.
    drop(stream);
    if !pending.is_empty() {
        processor.insert_audio_chunk(&pending)?;
        let output = processor.process()?;
        render(
            &output.committed,
            output.tentative,
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

/// Convert a seconds-valued timestamp to centiseconds (10 ms units).
fn to_cs(sec: f64) -> i64 {
    (sec * 100.0).round() as i64
}

/// Convert any cpal sample type to f32 and average all channels into
/// one mono frame. Channel count of 0 should not happen in practice.
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

/// Print just-committed words as a new line; redraw the tentative
/// overlay underneath so the user always sees both: a stable history
/// above and a moving prediction below. Committed lines use the
/// `[<start> - <end>]: <text>` centisecond format, taking the start
/// of the first and end of the last word in the batch.
fn render(committed: &[Word], tentative: Vec<Word>, sep: &str, tentative_visible: &mut bool) {
    if let (Some(first), Some(last)) = (committed.first(), committed.last()) {
        // Erase the tentative line first so the new committed line
        // lands on its own row instead of being prepended to the
        // overlay text.
        if *tentative_visible {
            eprint!("\r\x1b[K");
            *tentative_visible = false;
        }
        let joined = join_words(committed, sep);
        println!("[{} - {}]: {}", to_cs(first.start), to_cs(last.end), joined);
        let _ = std::io::stdout().flush();
    }

    // Re-draw tentative hypothesis on stderr so it stays visually
    // distinct from the committed stdout history.
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
/// adequate for ASR: speech energy lives below 4 kHz and Whisper
/// expects a 16 kHz log-mel spectrogram, so anti-aliasing is taken
/// care of by the model's own front-end. Keeps one sample of state
/// across calls so the interpolation is continuous at chunk
/// boundaries.
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
