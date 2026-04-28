//! Real-time microphone transcription with LocalAgreement-2 streaming.
//!
//! This example intentionally owns the microphone input layer. The crate handles
//! model loading, VAD, audio normalization helpers, resampling, ASR chunking,
//! and LocalAgreement processing through `AsrPipeline`.
//!
//! Usage:
//!     cargo run --release --example streaming_mic -- \
//!         <model.bin> [language=auto] [vad-model.bin]

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use yamabiko_whisper::{
    AsrPipeline, AudioInputConfig, AudioSample, OnlineAsrModel, SAMPLE_RATE, VadConfig, VadModel,
    Word, downmix_interleaved, install_log_hooks,
};

/// Run a Whisper pass once we have at least this many seconds of new audio.
const MIN_CHUNK_SEC: f64 = 1.0;
/// Target sampling rate for whisper.cpp (fixed by the model).
const TARGET_SR: u32 = SAMPLE_RATE as u32;

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
    let stream = match sample_format {
        SampleFormat::F32 => {
            build_input_stream::<f32>(&device, &stream_config, channels, tx.clone())?
        }
        SampleFormat::F64 => {
            build_input_stream::<f64>(&device, &stream_config, channels, tx.clone())?
        }
        SampleFormat::I8 => {
            build_input_stream::<i8>(&device, &stream_config, channels, tx.clone())?
        }
        SampleFormat::I16 => {
            build_input_stream::<i16>(&device, &stream_config, channels, tx.clone())?
        }
        SampleFormat::I32 => {
            build_input_stream::<i32>(&device, &stream_config, channels, tx.clone())?
        }
        SampleFormat::I64 => {
            build_input_stream::<i64>(&device, &stream_config, channels, tx.clone())?
        }
        SampleFormat::U8 => {
            build_input_stream::<u8>(&device, &stream_config, channels, tx.clone())?
        }
        SampleFormat::U16 => {
            build_input_stream::<u16>(&device, &stream_config, channels, tx.clone())?
        }
        SampleFormat::U32 => {
            build_input_stream::<u32>(&device, &stream_config, channels, tx.clone())?
        }
        SampleFormat::U64 => {
            build_input_stream::<u64>(&device, &stream_config, channels, tx.clone())?
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

    let mut pipeline = AsrPipeline::new(
        processor,
        AudioInputConfig::new(input_sr, 1).with_process_interval_sec(MIN_CHUNK_SEC),
    )?;
    let mut tentative_visible = false;

    eprintln!("speak now (Ctrl-C to stop)...");
    while running.load(Ordering::SeqCst) {
        if let Ok(chunk) = rx.recv_timeout(Duration::from_millis(100)) {
            push_audio_chunk(&mut pipeline, &chunk, &mut tentative_visible)?;
        }
        while let Ok(chunk) = rx.try_recv() {
            push_audio_chunk(&mut pipeline, &chunk, &mut tentative_visible)?;
        }
    }

    drop(stream);
    let output = pipeline.finish()?;
    render(
        &output.committed,
        output.tentative,
        pipeline.sep(),
        &mut tentative_visible,
    );

    Ok(())
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

fn push_audio_chunk(
    pipeline: &mut AsrPipeline,
    chunk: &[f32],
    tentative_visible: &mut bool,
) -> Result<()> {
    if let Some(output) = pipeline.push_mono(chunk)? {
        render(
            &output.committed,
            output.tentative,
            pipeline.sep(),
            tentative_visible,
        );
    }
    Ok(())
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
