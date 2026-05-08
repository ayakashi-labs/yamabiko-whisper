# yamabiko-whisper

[日本語](README.ja.md)

Low-latency streaming speech recognition for Rust, built on
[`whisper-rs`](https://crates.io/crates/whisper-rs), whisper.cpp, Silero VAD,
and the LocalAgreement-2 policy from streaming Whisper research.

Since 0.2, the supported usage path requires VAD. Load a GGML Whisper model and
a GGML Silero VAD model, create a VAD-backed processor, and feed audio through
`AsrPipeline`. The pipeline normalizes samples, downmixes channels, resamples
to 16 kHz, chunks audio, and returns committed words plus a tentative live
hypothesis.

> Status: this crate is currently validated primarily with Japanese speech.
> Other languages should work through Whisper's language support, but they have
> not been tested as thoroughly. The crate is still pre-1.0, and breaking API or
> behavior changes may continue while the streaming pipeline is refined.

## Install

```toml
[dependencies]
yamabiko-whisper = { version = "0.2", features = ["vulkan"] } # or ["cuda"]
```

On Windows, the default `whisper-rs` backend can be extremely slow in optimized
production builds. For production use, prefer an accelerated whisper.cpp backend
such as Vulkan or CUDA. This crate exposes the backend features provided by
`whisper-rs`, including `cuda`, `vulkan`, `metal`, `coreml`, `hipblas`,
`intel-sycl`, `openblas`, and `openmp`.

Model files are not bundled. Provide:

- a GGML-format Whisper model compatible with whisper.cpp
- a GGML-format Silero VAD model

## Start With The Microphone Example

The recommended reference implementation is
[`examples/streaming_mic.rs`](https://github.com/ayakashi-labs/yamabiko-whisper/blob/main/examples/streaming_mic.rs).
It shows the current flow: load both models, create a VAD-backed processor,
capture the default microphone with `cpal`, feed `AsrPipeline`, print
committed words, and render a tentative line.

```bash
cargo run --release --features vulkan --example streaming_mic -- <model.bin> <language> <vad-model.bin>
```

For language autodetection, pass `auto` explicitly:

```bash
cargo run --release --features vulkan --example streaming_mic -- ggml-large-v3-turbo-q5_0.bin auto ggml-silero-v5.1.2.bin
```

If your application already owns the audio input layer, mirror the example's ASR
side: create `OnlineAsrModel`, create `VadModel`, call
`create_processor_with_vad`, then wrap the processor in `AsrPipeline`.

## Feature Flags

By default, the crate builds the CPU backend exposed by `whisper-rs`.

On Windows, that default backend may be too slow for production speech
recognition. Use `vulkan` where possible, or `cuda` if your deployment targets
NVIDIA GPUs.

```toml
[dependencies]
yamabiko-whisper = { version = "0.2", features = ["vulkan"] }
# or
yamabiko-whisper = { version = "0.2", features = ["cuda"] }
```

Available pass-through features are `cuda`, `vulkan`, `metal`, `coreml`,
`hipblas`, `intel-sycl`, `openblas`, and `openmp`. They require the native
dependencies needed by `whisper-rs` and whisper.cpp for the selected backend
(for example, Vulkan SDK for `vulkan`, CUDA Toolkit for `cuda`, plus CMake).

On Windows, `vulkan` builds can hit CMake path length limits inside
`whisper-rs-sys`. If that happens, create a local `.cargo/config.toml` to use a
short target directory:

```toml
[build]
target-dir = "C:\\t"
```

When a GPU backend is compiled in, `BackendConfig::default()` follows the
`whisper-rs` default for GPU use. Use `OnlineAsrModel::load_with_backend` to
select a device or force CPU execution:

```rust
use yamabiko_whisper::{BackendConfig, OnlineAsrModel};

let backend = BackendConfig::default().with_gpu_device(1);
let model = OnlineAsrModel::load_with_backend("ggml-large-v3-turbo.bin", backend)?;
```
