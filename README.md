# yamabiko-whisper

[日本語](README.ja.md)

Low-latency streaming speech recognition for Rust, built on
[`whisper-rs`](https://crates.io/crates/whisper-rs) and the
LocalAgreement-2 policy from whisper streaming research.

The crate hides `whisper-rs` from your public code. Load a GGML Whisper
model once, create one or more processors, feed 16 kHz mono `f32` PCM
chunks, and receive only the words that are stable enough to commit.

> Note: This crate is currently tested primarily with Japanese speech.
> Other languages should work through Whisper's language support, but
> they have not been validated as thoroughly.

## Quick Start

```toml
[dependencies]
yamabiko-whisper = "0.1"
```

```rust,no_run
use yamabiko_whisper::{OnlineAsrModel, SAMPLE_RATE};

fn main() -> Result<(), yamabiko_whisper::Error> {
    let model = OnlineAsrModel::load("ggml-base.en.bin")?;
    let mut asr = model.create_processor("en")?;

    // 16 kHz mono f32 PCM. Values should normally be in [-1.0, 1.0].
    let pcm: Vec<f32> = vec![0.0; SAMPLE_RATE];

    asr.insert_audio_chunk(&pcm)?;
    let output = asr.process()?;
    for word in output.committed {
        println!("{:.2}-{:.2}: {}", word.start, word.end, word.text);
    }

    for word in asr.finish() {
        println!("{:.2}-{:.2}: {}", word.start, word.end, word.text);
    }

    Ok(())
}
```

Use `"auto"` as the language to let Whisper detect the language, or pass
a Whisper language code such as `"en"` or `"ja"`.

The first `process` call often returns no committed words. LocalAgreement-2
commits a word after it appears in the same prefix position in two
consecutive hypotheses. Use `output.tentative` for a live preview line.

## Input Audio

`OnlineAsrProcessor::insert_audio_chunk` expects:

- 16,000 Hz sample rate
- mono channel layout
- `f32` PCM samples
- normalized audio, usually in the `[-1.0, 1.0]` range

If your input comes from a microphone, file, or network stream at another
sample rate, resample it before calling `insert_audio_chunk`.

## VAD

Silero VAD is optional. It skips silent chunks and resets the Whisper
decoder state after a configurable silence window.

```rust,no_run
use yamabiko_whisper::{OnlineAsrModel, VadConfig, VadModel};

fn main() -> Result<(), yamabiko_whisper::Error> {
    let model = OnlineAsrModel::load("ggml-base.en.bin")?;
    let vad_model = VadModel::load_with_config(
        "ggml-silero-v5.1.2.bin",
        VadConfig::default(),
    )?;
    let mut asr = model.create_processor_with_vad("en", &vad_model)?;

    asr.insert_audio_chunk(&vec![0.0; 16_000])?;
    let output = asr.process()?;
    drop(output);

    Ok(())
}
```

When VAD closes an utterance, `process` flushes the current tentative
words and returns `ProcessOutput { finalized_by_vad: true, .. }`, so
callers can distinguish that boundary from a normal LocalAgreement
commit.

## Configuration

For one-off use, `OnlineAsrProcessor::from_model_path` and
`from_model_path_with_vad` remain available. For applications that create
more than one stream, prefer loading `OnlineAsrModel` once and creating
processors from it.

```rust,no_run
use yamabiko_whisper::{DecodingStrategy, OnlineAsrConfig, OnlineAsrModel};

fn main() -> Result<(), yamabiko_whisper::Error> {
    let model = OnlineAsrModel::load("ggml-base.en.bin")?;
    let config = OnlineAsrConfig::new("en")
        .with_n_threads(4)
        .with_buffer_trimming_sec(10.0)
        .with_prompt_char_budget(300)
        .with_decoding_strategy(DecodingStrategy::BeamSearch {
            beam_size: 3,
            patience: -1.0,
        });

    let _asr = model.create_processor_with_config(config)?;
    Ok(())
}
```

## Example: Microphone Streaming

The repository includes a microphone example using `cpal`.

```bash
cargo run --release --example streaming_mic -- <model.bin> [language=auto]
```

With VAD:

```bash
cargo run --release --example streaming_mic -- <model.bin> [language=auto] <vad-model.bin>
```

The example captures the default input device, downmixes to mono,
resamples to 16 kHz, prints committed words to stdout, and renders the
tentative hypothesis on stderr.

## Features

By default, the crate builds the CPU backend exposed by `whisper-rs`.

Enable Vulkan acceleration with:

```toml
[dependencies]
yamabiko-whisper = { version = "0.1", features = ["vulkan"] }
```

Building with `vulkan` requires the native dependencies needed by
`whisper-rs` and whisper.cpp, including a working Vulkan SDK and CMake.

## Model Files

Use GGML-format Whisper models compatible with whisper.cpp. The VAD entry
point expects a GGML Silero VAD model file.

Model files are not bundled with the crate.

## Development

Useful commands:

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo rustdoc --lib -- -D warnings
cargo package --allow-dirty
```

The repository keeps `.cargo/config.toml` locally to shorten the build
target path on Windows when Vulkan shader generation is enabled. That
local configuration is excluded from the published crate.
