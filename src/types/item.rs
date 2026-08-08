//! Request items: the multimodal unit every inference endpoint consumes.

use std::path::{Path, PathBuf};

use rmpv::Value as MsgValue;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::media::{self, Samples};

/// An image, in whatever form the caller already has it.
#[derive(Debug, Clone, PartialEq)]
enum ImageData {
    /// Already encoded; the format is sniffed from the bytes.
    Encoded(Vec<u8>),
    /// Read from disk at send time, then sniffed.
    Path(PathBuf),
    /// Decoded pixels, re-encoded as JPEG at send time.
    Decoded(Box<image::DynamicImage>),
}

/// One image attached to an [`Item`].
#[derive(Debug, Clone, PartialEq)]
pub struct ImageInput {
    data: ImageData,
    format: Option<String>,
}

impl ImageInput {
    /// An image that is already encoded. The format is detected from its magic bytes.
    pub fn bytes(data: impl Into<Vec<u8>>) -> Self {
        Self {
            data: ImageData::Encoded(data.into()),
            format: None,
        }
    }

    /// An encoded image read from disk when the request is sent.
    pub fn path(path: impl Into<PathBuf>) -> Self {
        Self {
            data: ImageData::Path(path.into()),
            format: None,
        }
    }

    /// Decoded pixels, re-encoded as JPEG when the request is sent.
    pub fn decoded(image: image::DynamicImage) -> Self {
        Self {
            data: ImageData::Decoded(Box::new(image)),
            format: None,
        }
    }

    /// Declare the format explicitly. A declaration that contradicts the bytes is an error.
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    pub(crate) fn resolve(&self) -> Result<(Vec<u8>, String)> {
        let encoded = match &self.data {
            ImageData::Encoded(data) => data.clone(),
            ImageData::Path(path) => read_file(path)?,
            ImageData::Decoded(image) => media::encode_jpeg(image)?,
        };
        let detected = media::detect_image_format(&encoded)?;
        if let Some(declared) = &self.format {
            let declared = media::canonical_format(declared)?;
            if declared != detected {
                return Err(Error::invalid(format!(
                    "Image format mismatch: declared {declared:?}, detected {detected:?}"
                )));
            }
        }
        Ok((encoded, detected.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum AudioData {
    Encoded(Vec<u8>),
    Path(PathBuf),
    Waveform {
        samples: Samples,
        channels: u16,
        sample_rate: u32,
    },
}

/// Audio attached to an [`Item`].
#[derive(Debug, Clone, PartialEq)]
pub struct AudioInput {
    data: AudioData,
    format: Option<String>,
}

impl AudioInput {
    /// Audio that is already in a container the server understands.
    pub fn bytes(data: impl Into<Vec<u8>>) -> Self {
        Self {
            data: AudioData::Encoded(data.into()),
            format: None,
        }
    }

    /// Encoded audio read from disk when the request is sent; the format comes from the
    /// filename.
    pub fn path(path: impl Into<PathBuf>) -> Self {
        Self {
            data: AudioData::Path(path.into()),
            format: None,
        }
    }

    /// A raw waveform, wrapped in a 16-bit PCM WAV container when the request is sent.
    pub fn waveform(samples: Samples, channels: u16, sample_rate: u32) -> Self {
        Self {
            data: AudioData::Waveform {
                samples,
                channels,
                sample_rate,
            },
            format: None,
        }
    }

    /// Declare the container format explicitly.
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    pub(crate) fn resolve(&self) -> Result<(Vec<u8>, Option<String>, Option<u32>)> {
        match &self.data {
            AudioData::Encoded(data) => Ok((data.clone(), self.format.clone(), None)),
            AudioData::Path(path) => {
                let inferred = media::infer_audio_format(path).map(str::to_string);
                Ok((read_file(path)?, self.format.clone().or(inferred), None))
            }
            AudioData::Waveform {
                samples,
                channels,
                sample_rate,
            } => Ok((
                media::encode_wav(samples, *channels, *sample_rate)?,
                Some(self.format.clone().unwrap_or_else(|| "wav".to_string())),
                Some(*sample_rate),
            )),
        }
    }
}

/// A document or video attached to an [`Item`]: bytes plus an optional format token.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryInput {
    data: BinaryData,
    format: Option<String>,
    kind: BinaryKind,
}

#[derive(Debug, Clone, PartialEq)]
enum BinaryData {
    Encoded(Vec<u8>),
    Path(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryKind {
    Document,
    Video,
}

impl BinaryInput {
    /// A document held in memory.
    pub fn document_bytes(data: impl Into<Vec<u8>>) -> Self {
        Self {
            data: BinaryData::Encoded(data.into()),
            format: None,
            kind: BinaryKind::Document,
        }
    }

    /// A document read from disk; the format comes from the filename.
    pub fn document_path(path: impl Into<PathBuf>) -> Self {
        Self {
            data: BinaryData::Path(path.into()),
            format: None,
            kind: BinaryKind::Document,
        }
    }

    /// A video held in memory.
    pub fn video_bytes(data: impl Into<Vec<u8>>) -> Self {
        Self {
            data: BinaryData::Encoded(data.into()),
            format: None,
            kind: BinaryKind::Video,
        }
    }

    /// A video read from disk; the format comes from the filename.
    pub fn video_path(path: impl Into<PathBuf>) -> Self {
        Self {
            data: BinaryData::Path(path.into()),
            format: None,
            kind: BinaryKind::Video,
        }
    }

    /// Declare the format explicitly rather than inferring it from the filename.
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    pub(crate) fn resolve(&self) -> Result<(Vec<u8>, Option<String>)> {
        match &self.data {
            BinaryData::Encoded(data) => Ok((data.clone(), self.format.clone())),
            BinaryData::Path(path) => {
                let inferred = match self.kind {
                    BinaryKind::Document => media::infer_document_format(path),
                    BinaryKind::Video => media::infer_video_format(path),
                }
                .map(str::to_string);
                Ok((read_file(path)?, self.format.clone().or(inferred)))
            }
        }
    }
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|err| {
        Error::Io(std::io::Error::new(
            err.kind(),
            format!("could not read {}: {err}", path.display()),
        ))
    })
}

/// One unit of input: text, images, audio, video or a document, in any combination the
/// model accepts.
#[derive(Debug, Clone, Default, PartialEq)]
#[allow(missing_docs)]
pub struct Item {
    /// Caller-chosen identifier, echoed back on the matching result.
    pub id: Option<String>,
    pub text: Option<String>,
    pub images: Vec<ImageInput>,
    pub audio: Option<AudioInput>,
    pub video: Option<BinaryInput>,
    pub document: Option<BinaryInput>,
    /// Opaque metadata passed through to the model adapter.
    pub metadata: Option<Value>,
}

impl Item {
    /// An empty item.
    pub fn new() -> Self {
        Self::default()
    }

    /// A text-only item.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            ..Self::default()
        }
    }

    /// An item holding a single image.
    pub fn image(image: ImageInput) -> Self {
        Self {
            images: vec![image],
            ..Self::default()
        }
    }

    /// Set the identifier echoed back on the result.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set or replace the text.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Append an image.
    pub fn with_image(mut self, image: ImageInput) -> Self {
        self.images.push(image);
        self
    }

    /// Attach audio.
    pub fn with_audio(mut self, audio: AudioInput) -> Self {
        self.audio = Some(audio);
        self
    }

    /// Attach a video.
    pub fn with_video(mut self, video: BinaryInput) -> Self {
        self.video = Some(video);
        self
    }

    /// Attach a document.
    pub fn with_document(mut self, document: BinaryInput) -> Self {
        self.document = Some(document);
        self
    }

    /// Attach adapter metadata.
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Encode to the msgpack shape, resolving every attachment to bytes.
    ///
    /// Only fields the caller set are emitted: `/v1` is additive, so an absent key and a
    /// null one are not the same thing.
    pub(crate) fn to_msgpack(&self) -> Result<MsgValue> {
        let mut fields: Vec<(MsgValue, MsgValue)> = Vec::new();

        if let Some(id) = &self.id {
            fields.push((MsgValue::from("id"), MsgValue::from(id.as_str())));
        }
        if let Some(text) = &self.text {
            fields.push((MsgValue::from("text"), MsgValue::from(text.as_str())));
        }
        if !self.images.is_empty() {
            let mut images = Vec::with_capacity(self.images.len());
            for image in &self.images {
                let (data, format) = image.resolve()?;
                images.push(MsgValue::Map(vec![
                    (MsgValue::from("data"), MsgValue::Binary(data)),
                    (MsgValue::from("format"), MsgValue::from(format)),
                ]));
            }
            fields.push((MsgValue::from("images"), MsgValue::Array(images)));
        }
        if let Some(audio) = &self.audio {
            let (data, format, sample_rate) = audio.resolve()?;
            fields.push((
                MsgValue::from("audio"),
                MsgValue::Map(vec![
                    (MsgValue::from("data"), MsgValue::Binary(data)),
                    (MsgValue::from("format"), optional_str(format)),
                    (
                        MsgValue::from("sample_rate"),
                        sample_rate.map_or(MsgValue::Nil, |rate| MsgValue::from(u64::from(rate))),
                    ),
                ]),
            ));
        }
        if let Some(video) = &self.video {
            let (data, format) = video.resolve()?;
            fields.push((
                MsgValue::from("video"),
                MsgValue::Map(vec![
                    (MsgValue::from("data"), MsgValue::Binary(data)),
                    (MsgValue::from("format"), optional_str(format)),
                ]),
            ));
        }
        if let Some(document) = &self.document {
            let (data, format) = document.resolve()?;
            fields.push((
                MsgValue::from("document"),
                MsgValue::Map(vec![
                    (MsgValue::from("data"), MsgValue::Binary(data)),
                    (MsgValue::from("format"), optional_str(format)),
                ]),
            ));
        }
        if let Some(metadata) = &self.metadata {
            fields.push((MsgValue::from("metadata"), json_to_msgpack(metadata)));
        }
        Ok(MsgValue::Map(fields))
    }
}

fn optional_str(value: Option<String>) -> MsgValue {
    value.map_or(MsgValue::Nil, MsgValue::from)
}

/// Translate a JSON value into its msgpack equivalent.
pub(crate) fn json_to_msgpack(value: &Value) -> MsgValue {
    match value {
        Value::Null => MsgValue::Nil,
        Value::Bool(flag) => MsgValue::Boolean(*flag),
        Value::Number(number) => number.as_i64().map_or_else(
            || {
                number.as_u64().map_or_else(
                    || MsgValue::from(number.as_f64().unwrap_or(0.0)),
                    MsgValue::from,
                )
            },
            MsgValue::from,
        ),
        Value::String(text) => MsgValue::from(text.as_str()),
        Value::Array(items) => MsgValue::Array(items.iter().map(json_to_msgpack).collect()),
        Value::Object(entries) => MsgValue::Map(
            entries
                .iter()
                .map(|(key, value)| (MsgValue::from(key.as_str()), json_to_msgpack(value)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    // These assertions are about exact values, so exact comparison is the point.
    #![allow(clippy::float_cmp)]

    use super::*;
    use serde_json::json;

    fn field<'a>(value: &'a MsgValue, name: &str) -> Option<&'a MsgValue> {
        match value {
            MsgValue::Map(entries) => entries
                .iter()
                .find(|(key, _)| key.as_str() == Some(name))
                .map(|(_, value)| value),
            _ => None,
        }
    }

    fn png_bytes() -> Vec<u8> {
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let mut buffer = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut buffer, image::ImageFormat::Png)
            .unwrap();
        buffer.into_inner()
    }

    #[test]
    fn a_text_item_emits_only_text() {
        let wire = Item::text("hello").to_msgpack().unwrap();
        assert_eq!(field(&wire, "text").unwrap().as_str(), Some("hello"));
        assert!(field(&wire, "images").is_none());
        assert!(field(&wire, "id").is_none());
    }

    #[test]
    fn encoded_images_pass_through_with_a_detected_format() {
        let png = png_bytes();
        let wire = Item::image(ImageInput::bytes(png.clone()))
            .to_msgpack()
            .unwrap();
        let images = field(&wire, "images").unwrap();
        let MsgValue::Array(images) = images else {
            panic!("expected an array")
        };
        assert_eq!(field(&images[0], "format").unwrap().as_str(), Some("png"));
        // The bytes are carried verbatim: an already-encoded image is never re-encoded.
        assert_eq!(
            field(&images[0], "data").unwrap(),
            &MsgValue::Binary(png),
            "image bytes were rewritten"
        );
    }

    #[test]
    fn decoded_images_become_jpeg() {
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let (data, format) = ImageInput::decoded(image).resolve().unwrap();
        assert_eq!(format, "jpeg");
        assert_eq!(media::detect_image_format(&data).unwrap(), "jpeg");
    }

    #[test]
    fn a_declared_format_that_contradicts_the_bytes_is_rejected() {
        let err = ImageInput::bytes(png_bytes())
            .format("jpeg")
            .resolve()
            .unwrap_err();
        assert!(err.to_string().contains("mismatch"), "{err}");
        // The alias still matches its canonical form.
        assert!(
            ImageInput::bytes(png_bytes())
                .format("PNG")
                .resolve()
                .is_ok()
        );
    }

    #[test]
    fn waveforms_are_wrapped_in_wav() {
        let audio = AudioInput::waveform(Samples::F32(vec![0.0, 0.5, -0.5, 0.0]), 1, 16_000);
        let (data, format, sample_rate) = audio.resolve().unwrap();
        assert_eq!(format.as_deref(), Some("wav"));
        assert_eq!(sample_rate, Some(16_000));
        assert_eq!(&data[..4], b"RIFF");
    }

    #[test]
    fn document_format_is_inferred_from_the_path_but_never_overrides_a_declaration() {
        let dir = std::env::temp_dir().join("sie-sdk-item-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.pdf");
        std::fs::write(&path, b"%PDF-1.4").unwrap();

        let (data, format) = BinaryInput::document_path(&path).resolve().unwrap();
        assert_eq!(data, b"%PDF-1.4");
        assert_eq!(format.as_deref(), Some("pdf"));

        let (_, declared) = BinaryInput::document_path(&path)
            .format("txt")
            .resolve()
            .unwrap();
        assert_eq!(declared.as_deref(), Some("txt"));

        // In-memory documents carry no format unless the caller declares one.
        assert_eq!(
            BinaryInput::document_bytes(b"raw".to_vec())
                .resolve()
                .unwrap()
                .1,
            None
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_missing_file_names_itself_in_the_error() {
        let err = ImageInput::path("/nonexistent/sie-sdk/image.png")
            .resolve()
            .unwrap_err();
        assert!(
            err.to_string().contains("/nonexistent/sie-sdk/image.png"),
            "{err}"
        );
    }

    #[test]
    fn metadata_translates_into_msgpack() {
        let wire = Item::text("x")
            .with_id("doc-1")
            .with_metadata(
                json!({"source": "web", "rank": 3, "tags": ["a"], "keep": null, "score": 1.5}),
            )
            .to_msgpack()
            .unwrap();
        assert_eq!(field(&wire, "id").unwrap().as_str(), Some("doc-1"));
        let metadata = field(&wire, "metadata").unwrap();
        assert_eq!(field(metadata, "source").unwrap().as_str(), Some("web"));
        assert_eq!(field(metadata, "rank").unwrap().as_i64(), Some(3));
        assert_eq!(field(metadata, "score").unwrap().as_f64(), Some(1.5));
        assert_eq!(field(metadata, "keep").unwrap(), &MsgValue::Nil);
    }
}
