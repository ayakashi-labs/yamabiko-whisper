# yamabiko-whisper

[English](README.md)

Rust向けの低遅延ストリーミング音声認識クレートです。
[`whisper-rs`](https://crates.io/crates/whisper-rs)、whisper.cpp、Silero VAD、
ストリーミングWhisper研究で使われているLocalAgreement-2方式を利用しています。

0.2以降では、VADを使う構成を必須の利用導線とします。GGML形式のWhisperモデルと
GGML形式のSilero VADモデルを読み込み、VAD付きprocessorを作成して、`AsrPipeline`へ音声を渡します。
`AsrPipeline`はサンプル正規化、チャンネルのダウンミックス、16 kHzへのリサンプリング、チャンク化を行い、
確定単語とライブ表示用の未確定仮説を返します。

> 状態: 現時点では、主に日本語音声で検証しています。Whisperの言語対応により他の言語でも動作する想定ですが、
> 日本語ほど十分には検証していません。また、このcrateはまだpre-1.0であり、ストリーミング処理の改善に伴って
> 破壊的なAPI変更や挙動変更がしばらく続く可能性があります。

## インストール

```toml
[dependencies]
yamabiko-whisper = { version = "0.2", features = ["vulkan"] } # または ["cuda"]
```

Windowsでは、`whisper-rs`のデフォルトbackendを最適化済みの本番ビルドで使うと、音声認識が極端に遅くなることがあります。
本番用途では、VulkanやCUDAなどの高速化backendを使うことを推奨します。このcrateでは、`whisper-rs`が提供する
`cuda`、`vulkan`、`metal`、`coreml`、`hipblas`、`intel-sycl`、`openblas`、`openmp`
featureを公開しています。

モデルファイルは同梱されません。次の2つを用意してください。

- whisper.cppと互換性のあるGGML形式のWhisperモデル
- GGML形式のSilero VADモデル

## example

推奨する実装の参照先は[`examples/streaming_mic.rs`](https://github.com/ayakashi-labs/yamabiko-whisper/blob/main/examples/streaming_mic.rs)です。
このexampleでは、現在想定する流れとして、2つのモデル読み込み、VAD付きprocessorの作成、`cpal`による
デフォルトマイク入力、`AsrPipeline`への投入、確定単語の出力、未確定行の表示までを扱っています。

```bash
cargo run --release --features vulkan --example streaming_mic -- <model.bin> <language> <vad-model.bin>
```

言語を自動判定する場合は、`auto`を明示して渡します。

```bash
cargo run --release --features vulkan --example streaming_mic -- ggml-large-v3-turbo-q5_0.bin auto ggml-silero-v5.1.2.bin
```

アプリケーション側で音声入力層を持っている場合は、exampleのASR側を参考にしてください。
`OnlineAsrModel`と`VadModel`を作り、`create_processor_with_vad`を呼び、processorを`AsrPipeline`で包みます。

## 機能フラグ

デフォルトでは、`whisper-rs`が提供するCPUバックエンドでビルドします。

Windowsでは、このデフォルトbackendだと本番の音声認識が極端に遅い場合があります。可能であれば`vulkan`を使ってください。
NVIDIA GPU向けのデプロイでは、`cuda` featureを使えます。

```toml
[dependencies]
yamabiko-whisper = { version = "0.2", features = ["vulkan"] }
# または
yamabiko-whisper = { version = "0.2", features = ["cuda"] }
```

公開しているpass-through featureは、`cuda`、`vulkan`、`metal`、`coreml`、`hipblas`、`intel-sycl`、
`openblas`、`openmp`です。選択したbackendに応じて、`whisper-rs`とwhisper.cppが必要とする
ネイティブ依存関係を用意してください。例として、`vulkan`にはVulkan SDK、`cuda`にはCUDA Toolkitが必要です。
どちらもCMakeが必要です。

Windowsでは、`vulkan` build時に`whisper-rs-sys`内部のパスが長くなり、CMakeのパス長制限に当たる場合があります。
その場合は、ローカルの`.cargo/config.toml`で短いtarget directoryを指定してください。

```toml
[build]
target-dir = "C:\\t"
```

GPU backendを有効にした場合、`BackendConfig::default()`は`whisper-rs`のGPU利用デフォルトに従います。
デバイスIDを選ぶ、またはCPU実行へ戻す場合は、`OnlineAsrModel::load_with_backend`を使います。

```rust
use yamabiko_whisper::{BackendConfig, OnlineAsrModel};

let backend = BackendConfig::default().with_gpu_device(1);
let model = OnlineAsrModel::load_with_backend("ggml-large-v3-turbo.bin", backend)?;
```
