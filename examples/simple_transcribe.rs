use hound::WavReader;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn main() {
    let path_to_model = std::env::args().nth(1).unwrap();
    let path_to_audio = std::env::args().nth(2).unwrap();

    // load a context and model
    let ctx = WhisperContext::new_with_params(path_to_model, WhisperContextParameters::default())
        .expect("failed to load model");

    // create a params object
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0,
    });
    params.set_language(Some("auto"));

    // use available parallelism for CPU-side work
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    params.set_n_threads(n_threads);

    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_print_special(false);

    // load audio from a 16-bit PCM, 16KHz, mono WAV file
    let audio_data: Vec<f32> = WavReader::open(&path_to_audio)
        .expect("failed to open wav")
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    // now we can run the model
    let mut state = ctx.create_state().expect("failed to create state");
    state
        .full(params, &audio_data[..])
        .expect("failed to run model");

    // fetch the results
    for segment in state.as_iter() {
        println!(
            "[{} - {}]: {}",
            // note start and end timestamps are in centiseconds
            // (10s of milliseconds)
            segment.start_timestamp(),
            segment.end_timestamp(),
            // the Display impl for WhisperSegment will replace invalid UTF-8 with the Unicode replacement character
            segment
        );
    }
}
