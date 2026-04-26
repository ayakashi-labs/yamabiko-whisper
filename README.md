# local-agreement-whisper

[日本語](README.ja.md)

Low-latency streaming speech recognition for Rust, built on
[`whisper-rs`](https://crates.io/crates/whisper-rs) and the
LocalAgreement-2 policy from whisper streaming research.

The crate hides `whisper-rs` from your public code. Load a GGML Whisper
model, feed 16 kHz mono `f32` PCM chunks, and receive only the words that
are stable enough to commit.

> Note: This crate is currently tested primarily with Japanese speech.
> Other languages should work through Whisper's language support, but
> they have not been validated as thoroughly.

## Quick Start

```toml
[dependencies]
local-agreement-whisper = "0.1"
```

```rust,no_run
use local_agreement_whisper::OnlineAsrProcessor;

fn main() -> Result<(), local_agreement_whisper::Error> {
    let mut asr = OnlineAsrProcessor::from_model_path("ggml-base.en.bin", "en")?;

    // 16 kHz mono f32 PCM. Values should normally be in [-1.0, 1.0].
    let pcm: Vec<f32> = vec![0.0; 16_000];

    asr.insert_audio_chunk(&pcm)?;
    for word in asr.process_iter()? {
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
use local_agreement_whisper::{OnlineAsrProcessor, VadConfig};

fn main() -> Result<(), local_agreement_whisper::Error> {
    let mut asr = OnlineAsrProcessor::from_model_path_with_vad(
        "ggml-base.en.bin",
        "en",
        "ggml-silero-v5.1.2.bin",
        VadConfig::default(),
    )?;

    asr.insert_audio_chunk(&vec![0.0; 16_000])?;
    let committed = asr.process_iter()?;
    drop(committed);

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
local-agreement-whisper = { version = "0.1", features = ["vulkan"] }
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
