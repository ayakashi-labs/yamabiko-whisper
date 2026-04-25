use hound::{SampleFormat, WavReader};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

// 引数 / Args: <model.bin> <audio.wav>
fn main() {
    let model_path = std::env::args().nth(1).expect("model path required");
    let audio_path = std::env::args().nth(2).expect("audio path required");

    // 16kHz mono の WAV のみ受け付ける
    // Only 16kHz mono WAV is accepted.
    let mut reader = WavReader::open(&audio_path).expect("failed to open wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "sample rate must be 16kHz");
    assert_eq!(spec.channels, 1, "audio must be mono");

    // PCM/float サンプルを f32 [-1.0, 1.0] に正規化
    // Normalize PCM/float samples to f32 in [-1.0, 1.0].
    let audio: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
        (SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / i32::MAX as f32)
            .collect(),
        (SampleFormat::Float, 32) => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        _ => panic!("unsupported sample format"),
    };

    // Vulkan GPU バックエンドを使用
    // Use Vulkan GPU backend.
    let mut cparams = WhisperContextParameters::default();
    cparams.use_gpu(true).gpu_device(0);
    let ctx = WhisperContext::new_with_params(&model_path, cparams).expect("failed to load model");

    // "auto" 指定は multilingual モデルが必須
    // "auto" requires a multilingual model.
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0,
    });
    params.set_language(Some("auto"));
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    let mut state = ctx.create_state().expect("failed to create state");
    state.full(params, &audio[..]).expect("failed to run model");

    // タイムスタンプはセンチ秒 (10ms 単位)
    // Timestamps are in centiseconds (10ms units).
    for segment in state.as_iter() {
        println!(
            "[{} - {}]: {}",
            segment.start_timestamp(),
            segment.end_timestamp(),
            segment
        );
    }
}
