//! Real-time microphone transcription with LocalAgreement-2 streaming.
//!
//! This example intentionally owns the microphone input layer. The crate handles
//! model loading, VAD, audio normalization helpers, resampling, ASR chunking,
//! and LocalAgreement processing through `AsrPipeline`.
//!
//! Threading layout:
//!   cpal audio thread  --(audio_tx: mpsc<Vec<f32>>)-->  ASR thread
//!   ASR thread         --(event_tx: mpsc<AsrEvent>)-->  main thread (render)
//!
//! Splitting Whisper/VAD inference off the main thread keeps the rendering
//! loop responsive and prevents the audio mpsc from backing up while a long
//! Whisper pass is in flight.
//!
//! Usage:
//!     cargo run --release --example streaming_mic -- \
//!         <model.bin> [language=auto] [vad-model.bin]

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use yamabiko_whisper::{
    AsrPipeline, AudioInputConfig, AudioSample, OnlineAsrModel, ProcessOutput, SAMPLE_RATE,
    VadConfig, VadModel, Word, downmix_interleaved, install_log_hooks,
};

/// Run a Whisper pass once we have at least this many seconds of new audio.
const MIN_CHUNK_SEC: f64 = 1.0;
/// Target sampling rate for whisper.cpp (fixed by the model).
const TARGET_SR: u32 = SAMPLE_RATE as u32;

/// Output of the ASR worker thread, consumed by the main render loop.
enum AsrEvent {
    Process(ProcessOutput),
    Finish(ProcessOutput),
}

fn main() -> Result<()> {
    install_log_hooks();

    let mut args = std::env::args().skip(1);
    let model_path = args.next().ok_or_else(|| {
        anyhow!("usage: streaming_mic <model.bin> [language=auto] [vad-model.bin]")
    })?;
    let language = args.next().unwrap_or_else(|| "auto".to_string());
    let vad_model_path = args.next();

    eprintln!("loading model: {model_path}");
    let model = OnlineAsrModel::load(&model_path)?;
    let processor = match vad_model_path.as_deref() {
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
    let sep = processor.sep();

    let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>();
    let (stream, input_sr) = open_input_stream(audio_tx)?;
    stream.play().context("failed to start input stream")?;

    let pipeline_config =
        AudioInputConfig::new(input_sr, 1).with_process_interval_sec(MIN_CHUNK_SEC);

    let (event_tx, event_rx) = mpsc::channel::<AsrEvent>();
    let asr_handle = spawn_asr_worker(processor, pipeline_config, audio_rx, event_tx);

    let running = Arc::new(AtomicBool::new(true));
    {
        let running = running.clone();
        ctrlc::set_handler(move || running.store(false, Ordering::SeqCst))
            .context("failed to install Ctrl-C handler")?;
    }

    let mut tentative_visible = false;
    eprintln!("speak now (Ctrl-C to stop)...");
    'main_loop: while running.load(Ordering::SeqCst) {
        match event_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(AsrEvent::Process(out)) => {
                render(&out.committed, out.tentative, sep, &mut tentative_visible);
            }
            Ok(AsrEvent::Finish(out)) => {
                render(&out.committed, out.tentative, sep, &mut tentative_visible);
                break 'main_loop;
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break 'main_loop,
        }
    }

    // Stopping the cpal stream drops the audio_tx clone held by the callback,
    // which disconnects audio_rx and lets the ASR thread emit its final
    // `Finish` event before exiting.
    drop(stream);
    while let Ok(event) = event_rx.recv() {
        match event {
            AsrEvent::Process(out) | AsrEvent::Finish(out) => {
                render(&out.committed, out.tentative, sep, &mut tentative_visible);
            }
        }
    }

    asr_handle
        .join()
        .map_err(|_| anyhow!("ASR worker thread panicked"))??;
    Ok(())
}

fn spawn_asr_worker(
    processor: yamabiko_whisper::OnlineAsrProcessor,
    pipeline_config: AudioInputConfig,
    audio_rx: mpsc::Receiver<Vec<f32>>,
    event_tx: mpsc::Sender<AsrEvent>,
) -> thread::JoinHandle<Result<()>> {
    thread::Builder::new()
        .name("asr-worker".to_string())
        .spawn(move || -> Result<()> {
            let mut pipeline = AsrPipeline::new(processor, pipeline_config)?;
            while let Ok(chunk) = audio_rx.recv() {
                if let Some(output) = pipeline.push_mono(&chunk)? {
                    if event_tx.send(AsrEvent::Process(output)).is_err() {
                        return Ok(());
                    }
                }
            }
            let final_output = pipeline.finish()?;
            let _ = event_tx.send(AsrEvent::Finish(final_output));
            Ok(())
        })
        .expect("failed to spawn ASR worker thread")
}

/// Open the default input device and build a stream that forwards mono f32 chunks.
fn open_input_stream(tx: mpsc::Sender<Vec<f32>>) -> Result<(cpal::Stream, u32)> {
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

    // Dispatch on the device's sample format; cpal needs the concrete type at compile time.
    macro_rules! build {
        ($ty:ty) => {
            build_input_stream::<$ty>(&device, &stream_config, channels, tx)?
        };
    }
    let stream = match sample_format {
        SampleFormat::F32 => build!(f32),
        SampleFormat::F64 => build!(f64),
        SampleFormat::I8 => build!(i8),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I32 => build!(i32),
        SampleFormat::I64 => build!(i64),
        SampleFormat::U8 => build!(u8),
        SampleFormat::U16 => build!(u16),
        SampleFormat::U32 => build!(u32),
        SampleFormat::U64 => build!(u64),
        other => bail!("unsupported sample format: {:?}", other),
    };

    Ok((stream, input_sr))
}

fn build_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + AudioSample + Send + 'static,
{
    let err_fn = |err| eprintln!("stream error: {err}");
    Ok(device.build_input_stream::<T, _, _>(
        config,
        move |data, _| {
            let _ = tx.send(downmix_interleaved(data, channels));
        },
        err_fn,
        None,
    )?)
}

fn to_cs(sec: f64) -> i64 {
    (sec * 100.0).round() as i64
}

fn join_words(words: &[Word], sep: &str) -> String {
    let parts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    parts.join(sep)
}

fn render(committed: &[Word], tentative: Vec<Word>, sep: &str, tentative_visible: &mut bool) {
    if let (Some(first), Some(last)) = (committed.first(), committed.last()) {
        if *tentative_visible {
            eprint!("\r\x1b[K");
            *tentative_visible = false;
        }
        let joined = join_words(committed, sep);
        println!("[{} - {}]: {}", to_cs(first.start), to_cs(last.end), joined);
        let _ = std::io::stdout().flush();
    }

    if !tentative.is_empty() {
        let joined = join_words(&tentative, sep);
        eprint!("\r  ...(tentative): {joined}\x1b[K");
        let _ = std::io::stderr().flush();
        *tentative_visible = true;
    } else if *tentative_visible {
        eprint!("\r\x1b[K");
        let _ = std::io::stderr().flush();
        *tentative_visible = false;
    }
}
