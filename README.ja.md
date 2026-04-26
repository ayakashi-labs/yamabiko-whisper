# yamabiko-whisper

[English](README.md)

Rust向けの低遅延ストリーミング音声認識クレートです。
[`whisper-rs`](https://crates.io/crates/whisper-rs) と、ストリーミングWhisper研究で使われている
LocalAgreement-2方式を利用しています。

このクレートは、利用者のコードから`whisper-rs`を隠蔽します。GGML形式のWhisperモデルを一度読み込み、
1つ以上のprocessorを作成し、16 kHz monoの`f32` PCMチャンクを渡すと、安定したと判断された単語だけを
確定結果として受け取れます。

> 注意: 現時点では、主に日本語音声でテストしています。Whisperの言語対応により他の言語でも動作する想定ですが、
> 日本語ほど十分には検証していません。

## クイックスタート

```toml
[dependencies]
yamabiko-whisper = "0.1"
```

```rust,no_run
use yamabiko_whisper::{OnlineAsrModel, SAMPLE_RATE};

fn main() -> Result<(), yamabiko_whisper::Error> {
    let model = OnlineAsrModel::load("ggml-base.en.bin")?;
    let mut asr = model.create_processor("en")?;

    // 16 kHz mono f32 PCM。通常、値は [-1.0, 1.0] の範囲に正規化します。
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

Whisperに言語を自動判定させたい場合は`"auto"`を指定します。明示する場合は、`"en"`や`"ja"`などの
Whisper言語コードを渡してください。

最初の`process`呼び出しでは、確定単語が返らないことがよくあります。LocalAgreement-2は、
2回連続の仮説で同じprefix位置に出た単語を確定します。ライブプレビューには`output.tentative`を使ってください。

## 入力音声

`OnlineAsrProcessor::insert_audio_chunk`は、次の形式の音声を想定しています。

- サンプルレート: 16,000 Hz
- チャンネル: mono
- サンプル形式: `f32` PCM
- 音量範囲: 通常は`[-1.0, 1.0]`に正規化

マイク、音声ファイル、ネットワークストリームなどから別のサンプルレートで音声を受け取る場合は、
`insert_audio_chunk`を呼ぶ前に16 kHzへリサンプリングしてください。

## VAD

Silero VADは任意です。有効にすると、無音チャンクをスキップし、設定した無音時間を超えたあとに
Whisperデコーダ状態をリセットします。

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

VADが発話区間の終端を検出した場合、`process`は現在の未確定単語をflushし、
`ProcessOutput { finalized_by_vad: true, .. }`を返します。これにより、通常のLocalAgreement確定と
VAD境界での確定を呼び出し側で区別できます。

## 設定

一度だけ使う場合は、`OnlineAsrProcessor::from_model_path`と`from_model_path_with_vad`も引き続き利用できます。
複数ストリームを作るアプリでは、`OnlineAsrModel`を一度だけ読み込み、そこからprocessorを作る使い方を推奨します。

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

## Example: マイク入力のストリーミング

このリポジトリには、`cpal`を使ったマイク入力exampleが含まれています。

```bash
cargo run --release --example streaming_mic -- <model.bin> [language=auto]
```

VADも使う場合:

```bash
cargo run --release --example streaming_mic -- <model.bin> [language=auto] <vad-model.bin>
```

このexampleはデフォルト入力デバイスから音声を取得し、monoへダウンミックスして16 kHzへリサンプリングします。
確定した単語はstdoutへ出力し、未確定の仮説はstderrに表示します。

## Features

デフォルトでは、`whisper-rs`が提供するCPUバックエンドでビルドします。

Vulkanアクセラレーションを有効にするには、次のように`vulkan` featureを指定してください。

```toml
[dependencies]
yamabiko-whisper = { version = "0.1", features = ["vulkan"] }
```

`vulkan`を有効にしてビルドする場合は、`whisper-rs`とwhisper.cppが必要とするネイティブ依存関係が必要です。
Vulkan SDKとCMakeが利用できる環境を用意してください。

## モデルファイル

whisper.cppと互換性のあるGGML形式のWhisperモデルを使ってください。VAD用のエントリポイントでは、
GGML形式のSilero VADモデルファイルを想定しています。

モデルファイルはこのクレートには同梱されません。

## 開発

よく使うコマンド:

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo rustdoc --lib -- -D warnings
cargo package --allow-dirty
```

このリポジトリには、WindowsでVulkanシェーダ生成時のターゲットパスを短くするために
`.cargo/config.toml`を置いています。このローカル設定は、公開されるクレートには含めません。
