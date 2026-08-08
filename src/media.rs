//! Turning caller-supplied images, audio and documents into the bytes the wire carries.
//!
//! Format detection is by magic bytes and fails closed: an undetectable image is an error,
//! never a guess. Encoded inputs pass through untouched, so a JPEG the caller already has
//! is never re-encoded.

use std::io::Cursor;
use std::path::Path;

use crate::error::{Error, Result};

const JPEG_QUALITY: u8 = 95;
const MAX_FORMAT_LEN: usize = 32;

/// Normalise a caller-declared media format token.
pub(crate) fn canonical_format(value: &str) -> Result<String> {
    let invalid = || {
        Error::invalid(format!(
            "Image format must be a short ASCII media-format token, got {value:?}"
        ))
    };
    if value.is_empty() || value.len() > MAX_FORMAT_LEN {
        return Err(invalid());
    }
    if !value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'+' || b == b'-')
    {
        return Err(invalid());
    }
    let lowered = value.to_ascii_lowercase();
    Ok(match lowered.as_str() {
        "jpg" | "jpe" => "jpeg".to_string(),
        _ => lowered,
    })
}

/// Identify an encoded image by its magic bytes.
///
/// Formats without a signature the SDK recognises (HEIC and AVIF among them) are rejected
/// rather than mislabelled.
pub(crate) fn detect_image_format(data: &[u8]) -> Result<&'static str> {
    let format = if data.starts_with(&[0xff, 0xd8, 0xff]) {
        "jpeg"
    } else if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        "png"
    } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        "gif"
    } else if data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" {
        "webp"
    } else if data.starts_with(b"BM") {
        "bmp"
    } else if data.starts_with(b"II*\x00")
        || data.starts_with(b"MM\x00*")
        || data.starts_with(b"II+\x00")
        || data.starts_with(b"MM\x00+")
    {
        "tiff"
    } else {
        return Err(Error::invalid(
            "Could not detect encoded image format from bytes",
        ));
    };
    Ok(format)
}

/// Encode a decoded image as JPEG.
///
/// Greyscale stays greyscale; everything else becomes RGB, since JPEG has no alpha channel.
pub(crate) fn encode_jpeg(image: &image::DynamicImage) -> Result<Vec<u8>> {
    use image::ColorType;

    let converted = match image.color() {
        ColorType::L8 | ColorType::L16 => image::DynamicImage::ImageLuma8(image.to_luma8()),
        _ => image::DynamicImage::ImageRgb8(image.to_rgb8()),
    };
    let mut buffer = Cursor::new(Vec::new());
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, JPEG_QUALITY);
    encoder
        .encode_image(&converted)
        .map_err(|err| Error::invalid(format!("could not encode image as JPEG: {err}")))?;
    Ok(buffer.into_inner())
}

fn suffix(path: &Path) -> Option<String> {
    Some(path.extension()?.to_str()?.to_ascii_lowercase())
}

/// Audio container implied by a filename.
pub(crate) fn infer_audio_format(path: &Path) -> Option<&'static str> {
    Some(match suffix(path)?.as_str() {
        "flac" => "flac",
        "m4a" => "m4a",
        "mp3" => "mp3",
        "mp4" => "mp4",
        "mpeg" => "mpeg",
        "mpga" => "mpga",
        "ogg" => "ogg",
        "wav" => "wav",
        "webm" => "webm",
        _ => return None,
    })
}

/// Document type implied by a filename.
pub(crate) fn infer_document_format(path: &Path) -> Option<&'static str> {
    Some(match suffix(path)?.as_str() {
        "pdf" => "pdf",
        "docx" => "docx",
        "doc" => "doc",
        "html" | "htm" | "xhtml" => "html",
        "md" | "markdown" => "md",
        "txt" => "txt",
        "rtf" => "rtf",
        "odt" => "odt",
        "pptx" => "pptx",
        "xlsx" => "xlsx",
        "csv" => "csv",
        _ => return None,
    })
}

/// Video container implied by a filename.
pub(crate) fn infer_video_format(path: &Path) -> Option<&'static str> {
    Some(match suffix(path)?.as_str() {
        "mp4" => "mp4",
        "webm" => "webm",
        "mkv" => "mkv",
        "mov" => "mov",
        "avi" => "avi",
        _ => return None,
    })
}

/// One PCM sample buffer, in the precision the caller has.
#[derive(Debug, Clone, PartialEq)]
pub enum Samples {
    /// Normalised floats, nominally in `[-1.0, 1.0]`.
    F32(Vec<f32>),
    /// Signed 16-bit PCM, written through untouched.
    I16(Vec<i16>),
}

impl Samples {
    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::I16(values) => values.len(),
        }
    }

    /// Convert to signed 16-bit PCM.
    ///
    /// The scale is asymmetric because the i16 range is: `-1.0` maps to `-32768` and `1.0`
    /// to `32767`, so neither extreme clips.
    fn to_pcm16(&self) -> Result<Vec<i16>> {
        match self {
            Self::I16(values) => Ok(values.clone()),
            Self::F32(values) => values
                .iter()
                .map(|sample| {
                    if !sample.is_finite() {
                        return Err(Error::invalid("Audio samples must all be finite"));
                    }
                    let clamped = f64::from(*sample).clamp(-1.0, 1.0);
                    let scaled = clamped * if clamped < 0.0 { 32768.0 } else { 32767.0 };
                    // numpy rounds half to even; matching it keeps encoded audio identical
                    // to what the Python SDK produces for the same input.
                    Ok(scaled.round_ties_even().clamp(-32768.0, 32767.0) as i16)
                })
                .collect(),
        }
    }
}

/// Wrap raw samples in a WAV container.
///
/// Samples are frame-interleaved: for stereo, `[l0, r0, l1, r1, ...]`.
pub(crate) fn encode_wav(samples: &Samples, channels: u16, sample_rate: u32) -> Result<Vec<u8>> {
    if sample_rate == 0 {
        return Err(Error::invalid(
            "Audio sample_rate must be a positive integer",
        ));
    }
    if !(1..=2).contains(&channels) {
        return Err(Error::invalid(format!(
            "Audio must contain 1 or 2 channels, got {channels}"
        )));
    }
    if samples.len() == 0 {
        return Err(Error::invalid("Audio must not be empty"));
    }
    if !samples.len().is_multiple_of(usize::from(channels)) {
        return Err(Error::invalid(format!(
            "Audio has {} samples, which is not a whole number of {channels}-channel frames",
            samples.len()
        )));
    }

    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buffer = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut buffer, spec)
        .map_err(|err| Error::invalid(format!("could not start a WAV stream: {err}")))?;
    for sample in samples.to_pcm16()? {
        writer
            .write_sample(sample)
            .map_err(|err| Error::invalid(format!("could not write a WAV sample: {err}")))?;
    }
    writer
        .finalize()
        .map_err(|err| Error::invalid(format!("could not finalize the WAV stream: {err}")))?;
    Ok(buffer.into_inner())
}

#[cfg(test)]
mod tests {
    // These assertions are about exact values, so exact comparison is the point.
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn canonicalizes_format_aliases() {
        assert_eq!(canonical_format("JPG").unwrap(), "jpeg");
        assert_eq!(canonical_format("jpe").unwrap(), "jpeg");
        assert_eq!(canonical_format("PNG").unwrap(), "png");
        assert_eq!(canonical_format("image+xml").unwrap(), "image+xml");
    }

    #[test]
    fn rejects_format_tokens_that_are_not_short_ascii() {
        for value in ["", &"x".repeat(33), "png/", "png ", "påäng"] {
            assert!(canonical_format(value).is_err(), "{value:?}");
        }
    }

    #[test]
    fn sniffs_every_supported_signature() {
        let cases: [(&[u8], &str); 8] = [
            (&[0xff, 0xd8, 0xff, 0xe0], "jpeg"),
            (b"\x89PNG\r\n\x1a\n....", "png"),
            (b"GIF87a...", "gif"),
            (b"GIF89a...", "gif"),
            (b"RIFF\0\0\0\0WEBPVP8 ", "webp"),
            (b"BM......", "bmp"),
            (b"II*\x00....", "tiff"),
            (b"MM\x00+....", "tiff"),
        ];
        for (data, expected) in cases {
            assert_eq!(detect_image_format(data).unwrap(), expected);
        }
    }

    #[test]
    fn sniffing_fails_closed() {
        assert!(detect_image_format(b"").is_err());
        assert!(detect_image_format(b"\x00\x00\x00\x20ftypheic").is_err());
        // A truncated RIFF header is not a WebP.
        assert!(detect_image_format(b"RIFF\0\0\0\0WEB").is_err());
    }

    #[test]
    fn encodes_a_decoded_image_as_jpeg() {
        let image = image::DynamicImage::ImageRgba8(image::RgbaImage::new(4, 4));
        let bytes = encode_jpeg(&image).unwrap();
        assert_eq!(detect_image_format(&bytes).unwrap(), "jpeg");

        let grey = image::DynamicImage::ImageLuma8(image::GrayImage::new(4, 4));
        assert_eq!(
            detect_image_format(&encode_jpeg(&grey).unwrap()).unwrap(),
            "jpeg"
        );
    }

    #[test]
    fn infers_formats_from_suffixes() {
        assert_eq!(infer_audio_format(Path::new("/tmp/clip.WAV")), Some("wav"));
        assert_eq!(infer_audio_format(Path::new("/tmp/clip.aiff")), None);
        assert_eq!(
            infer_document_format(Path::new("report.Markdown")),
            Some("md")
        );
        assert_eq!(infer_document_format(Path::new("page.htm")), Some("html"));
        assert_eq!(infer_document_format(Path::new("noext")), None);
        assert_eq!(infer_video_format(Path::new("a.mp4")), Some("mp4"));
    }

    #[test]
    fn float_samples_use_the_asymmetric_pcm_scale() {
        let pcm = Samples::F32(vec![-1.0, 0.0, 1.0, 0.5]).to_pcm16().unwrap();
        assert_eq!(pcm, vec![-32768, 0, 32767, 16384]);
    }

    #[test]
    fn float_samples_are_clamped_and_must_be_finite() {
        assert_eq!(
            Samples::F32(vec![-2.0, 2.0]).to_pcm16().unwrap(),
            vec![-32768, 32767]
        );
        assert!(Samples::F32(vec![f32::NAN]).to_pcm16().is_err());
        assert!(Samples::F32(vec![f32::INFINITY]).to_pcm16().is_err());
    }

    #[test]
    fn wav_round_trips_through_hound() {
        let samples = Samples::I16(vec![1, -1, 100, -100]);
        let wav = encode_wav(&samples, 2, 16_000).unwrap();
        let mut reader = hound::WavReader::new(Cursor::new(wav)).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 2);
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(spec.bits_per_sample, 16);
        let decoded: Vec<i16> = reader
            .samples::<i16>()
            .map(std::result::Result::unwrap)
            .collect();
        assert_eq!(decoded, vec![1, -1, 100, -100]);
    }

    #[test]
    fn wav_rejects_impossible_geometry() {
        assert!(encode_wav(&Samples::I16(vec![1]), 1, 0).is_err());
        assert!(encode_wav(&Samples::I16(vec![1]), 3, 16_000).is_err());
        assert!(encode_wav(&Samples::I16(Vec::new()), 1, 16_000).is_err());
        // Three samples cannot be split into stereo frames.
        assert!(encode_wav(&Samples::I16(vec![1, 2, 3]), 2, 16_000).is_err());
    }
}
