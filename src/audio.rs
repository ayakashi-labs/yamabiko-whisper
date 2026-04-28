//! Audio input helpers for feeding [`OnlineAsrProcessor`](crate::OnlineAsrProcessor).
//!
//! These helpers keep source-specific audio plumbing out of application code:
//! callers can push interleaved samples from a microphone, file decoder, or
//! network stream at the source sample rate, and the pipeline converts them to
//! the 16 kHz mono PCM expected by Whisper.

use crate::error::Error;
use crate::online_asr::{OnlineAsrProcessor, ProcessOutput, SAMPLE_RATE};

/// Source audio format accepted by [`AsrPipeline`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioInputConfig {
    /// Source sample rate in Hz.
    pub sample_rate: u32,
    /// Number of interleaved source channels.
    pub channels: usize,
    /// Amount of new 16 kHz audio to accumulate before running Whisper.
    pub process_interval_sec: f64,
}

impl AudioInputConfig {
    /// Create an input config using the default 1 second processing interval.
    pub fn new(sample_rate: u32, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            process_interval_sec: 1.0,
        }
    }

    /// Set how much new audio is buffered before each ASR pass.
    pub fn with_process_interval_sec(mut self, process_interval_sec: f64) -> Self {
        self.process_interval_sec = process_interval_sec;
        self
    }

    fn validate(self) -> Result<Self, Error> {
        if self.sample_rate == 0 {
            return Err(Error::InvalidAudio(
                "input sample rate must be greater than zero".to_string(),
            ));
        }
        if self.channels == 0 {
            return Err(Error::InvalidAudio(
                "input channel count must be greater than zero".to_string(),
            ));
        }
        if !self.process_interval_sec.is_finite() || self.process_interval_sec <= 0.0 {
            return Err(Error::InvalidAudio(
                "process interval must be a positive finite number".to_string(),
            ));
        }
        Ok(self)
    }
}

/// Primitive audio sample types that can be normalized to `f32` PCM.
///
/// Integer samples are mapped to the usual `[-1.0, 1.0]` audio range. Floating
/// samples are passed through as `f32`.
pub trait AudioSample: Copy {
    /// Convert this sample to normalized `f32` PCM.
    fn to_f32_sample(self) -> f32;
}

impl AudioSample for f32 {
    fn to_f32_sample(self) -> f32 {
        self
    }
}

impl AudioSample for f64 {
    fn to_f32_sample(self) -> f32 {
        self as f32
    }
}

macro_rules! impl_signed_sample {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl AudioSample for $ty {
                fn to_f32_sample(self) -> f32 {
                    ((self as f64) / (<$ty>::MAX as f64)).clamp(-1.0, 1.0) as f32
                }
            }
        )+
    };
}

macro_rules! impl_unsigned_sample {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl AudioSample for $ty {
                fn to_f32_sample(self) -> f32 {
                    const MIDPOINT: f64 = (<$ty>::MAX as u128).div_ceil(2) as f64;
                    (((self as f64) - MIDPOINT) / MIDPOINT).clamp(-1.0, 1.0) as f32
                }
            }
        )+
    };
}

impl_signed_sample!(i8, i16, i32, i64);
impl_unsigned_sample!(u8, u16, u32, u64);

/// Downmix interleaved source audio to mono `f32` PCM.
///
/// Any trailing partial frame is ignored. A zero channel count returns an empty
/// vector; [`AsrPipeline::new`] rejects such a config before this helper is used.
pub fn downmix_interleaved<T>(samples: &[T], channels: usize) -> Vec<f32>
where
    T: AudioSample,
{
    if channels == 0 {
        return Vec::new();
    }
    if channels == 1 {
        return samples
            .iter()
            .map(|&sample| sample.to_f32_sample())
            .collect();
    }

    samples
        .chunks_exact(channels)
        .map(|frame| {
            let sum: f32 = frame.iter().map(|&sample| sample.to_f32_sample()).sum();
            sum / channels as f32
        })
        .collect()
}

/// Streaming linear-interpolation resampler.
///
/// The resampler keeps one sample of state across calls so interpolation remains
/// continuous at chunk boundaries. It is intentionally small and dependency-free;
/// applications that need studio-grade conversion can resample before calling
/// the pipeline.
#[derive(Clone, Debug)]
pub struct LinearResampler {
    ratio: f64,
    pos: f64,
    last: f32,
}

impl LinearResampler {
    /// Create a resampler from `input_sample_rate` to `output_sample_rate`.
    pub fn new(input_sample_rate: u32, output_sample_rate: u32) -> Result<Self, Error> {
        if input_sample_rate == 0 || output_sample_rate == 0 {
            return Err(Error::InvalidAudio(
                "sample rates must be greater than zero".to_string(),
            ));
        }

        Ok(Self {
            ratio: input_sample_rate as f64 / output_sample_rate as f64,
            pos: 0.0,
            last: 0.0,
        })
    }

    /// Convert a mono chunk to the configured output sample rate.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        if (self.ratio - 1.0).abs() < f64::EPSILON {
            self.last = *input.last().expect("input is not empty");
            return input.to_vec();
        }

        let mut output = Vec::with_capacity((input.len() as f64 / self.ratio) as usize + 1);
        let len = input.len() as f64;
        while self.pos < len {
            let i = self.pos.floor() as isize;
            let frac = (self.pos - self.pos.floor()) as f32;
            let next_idx = (i + 1) as usize;
            if next_idx >= input.len() {
                break;
            }

            let a = if i < 0 { self.last } else { input[i as usize] };
            let b = input[next_idx];
            output.push(a * (1.0 - frac) + b * frac);
            self.pos += self.ratio;
        }

        self.pos -= input.len() as f64;
        self.last = *input.last().expect("input is not empty");
        output
    }
}

/// High-level audio-to-ASR pipeline.
///
/// This wraps an [`OnlineAsrProcessor`] with source audio normalization,
/// downmixing, resampling to [`SAMPLE_RATE`], and process-interval buffering.
pub struct AsrPipeline {
    processor: OnlineAsrProcessor,
    resampler: LinearResampler,
    pending: Vec<f32>,
    channels: usize,
    min_samples: usize,
}

impl AsrPipeline {
    /// Build a pipeline for an existing processor and source audio format.
    pub fn new(processor: OnlineAsrProcessor, config: AudioInputConfig) -> Result<Self, Error> {
        let config = config.validate()?;
        let min_samples = (config.process_interval_sec * SAMPLE_RATE as f64)
            .round()
            .max(1.0) as usize;

        Ok(Self {
            processor,
            resampler: LinearResampler::new(config.sample_rate, SAMPLE_RATE as u32)?,
            pending: Vec::with_capacity(min_samples),
            channels: config.channels,
            min_samples,
        })
    }

    /// Push interleaved source samples into the pipeline.
    ///
    /// Returns `Some(output)` when enough new audio has accumulated to run one
    /// ASR pass; otherwise returns `None`.
    pub fn push_interleaved<T>(&mut self, samples: &[T]) -> Result<Option<ProcessOutput>, Error>
    where
        T: AudioSample,
    {
        let mono = downmix_interleaved(samples, self.channels);
        self.push_mono(&mono)
    }

    /// Push mono source samples into the pipeline.
    ///
    /// The samples are still interpreted at the input sample rate configured for
    /// this pipeline and will be resampled to 16 kHz if needed.
    pub fn push_mono(&mut self, samples: &[f32]) -> Result<Option<ProcessOutput>, Error> {
        self.pending.extend(self.resampler.process(samples));
        if self.pending.len() < self.min_samples {
            return Ok(None);
        }

        self.process_pending().map(Some)
    }

    /// Process any buffered audio even if the interval threshold was not met.
    pub fn flush_pending(&mut self) -> Result<Option<ProcessOutput>, Error> {
        if self.pending.is_empty() {
            return Ok(None);
        }

        self.process_pending().map(Some)
    }

    /// Flush pending audio and consume the processor's final tentative words.
    ///
    /// Returned words are placed in `committed`, with `tentative` always empty.
    pub fn finish(&mut self) -> Result<ProcessOutput, Error> {
        let mut output = self.flush_pending()?.unwrap_or_default();
        output.committed.extend(self.processor.finish());
        output.tentative.clear();
        Ok(output)
    }

    /// Access the wrapped processor.
    pub fn processor(&self) -> &OnlineAsrProcessor {
        &self.processor
    }

    /// Mutably access the wrapped processor.
    pub fn processor_mut(&mut self) -> &mut OnlineAsrProcessor {
        &mut self.processor
    }

    /// Word separator selected by the wrapped processor.
    pub fn sep(&self) -> &'static str {
        self.processor.sep()
    }

    fn process_pending(&mut self) -> Result<ProcessOutput, Error> {
        self.processor.insert_audio_chunk(&self.pending)?;
        self.pending.clear();
        self.processor.process()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmixes_stereo_i16_to_mono_f32() {
        let samples = [i16::MAX, 0, 0, i16::MIN];

        let mono = downmix_interleaved(&samples, 2);

        assert!((mono[0] - 0.5).abs() < 0.0001);
        assert!((mono[1] + 0.5).abs() < 0.0001);
    }

    #[test]
    fn downmix_ignores_partial_frame() {
        let samples = [1.0_f32, 3.0, 100.0];

        assert_eq!(downmix_interleaved(&samples, 2), vec![2.0]);
    }

    #[test]
    fn resampler_fast_path_keeps_same_rate_chunks() {
        let mut resampler = LinearResampler::new(16_000, 16_000).unwrap();
        let input = vec![0.0, 0.25, -0.25];

        assert_eq!(resampler.process(&input), input);
    }

    #[test]
    fn validates_pipeline_input_config() {
        assert!(AudioInputConfig::new(0, 1).validate().is_err());
        assert!(AudioInputConfig::new(16_000, 0).validate().is_err());
        assert!(
            AudioInputConfig::new(16_000, 1)
                .with_process_interval_sec(f64::NAN)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn signed_minimum_is_clamped_to_minus_one() {
        assert_eq!(i16::MIN.to_f32_sample(), -1.0);
    }
}
