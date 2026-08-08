//! Decoder for the `msgpack-numpy` tensor encoding.
//!
//! `/v1/encode`, `/v1/score`, `/v1/extract` and job result chunks are msgpack, and every
//! tensor inside them is a map with binary keys:
//!
//! ```text
//! {b"nd": true, b"type": "<f4", b"kind": b"", b"shape": [rows, cols], b"data": <raw bytes>}
//! ```
//!
//! `data` holds the raw buffer in C order, in the byte order the `type` string declares.

use rmpv::Value;

use crate::error::{Error, Result};

/// Element type of a numpy buffer, as declared by its dtype string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum Element {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl Element {
    fn width(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::F16 | Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }
}

/// Byte order declared by the dtype string's first character.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteOrder {
    Little,
    Big,
}

/// A decoded numpy buffer, still in its wire element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawArray {
    /// The dtype the server declared.
    pub element: Element,
    /// Dimensions, in C order.
    pub shape: Vec<usize>,
    order: ByteOrder,
    data: Vec<u8>,
}

fn parse_dtype(descr: &str) -> Result<(ByteOrder, Element)> {
    let (order_char, kind) = descr
        .split_at_checked(1)
        .ok_or_else(|| Error::decode(format!("empty numpy dtype descriptor: {descr:?}")))?;
    let order = match order_char {
        ">" => ByteOrder::Big,
        // "<" little, "|" not applicable (single-byte), "=" native. Every platform the SDK
        // runs on is little-endian, so native and not-applicable both mean little here.
        "<" | "|" | "=" => ByteOrder::Little,
        _ => {
            return Err(Error::decode(format!(
                "unsupported numpy byte-order marker in {descr:?}"
            )));
        }
    };
    let element = match kind {
        "f2" => Element::F16,
        "f4" => Element::F32,
        "f8" => Element::F64,
        "i1" => Element::I8,
        "i2" => Element::I16,
        "i4" => Element::I32,
        "i8" => Element::I64,
        "u1" => Element::U8,
        "u2" => Element::U16,
        "u4" => Element::U32,
        "u8" => Element::U64,
        other => return Err(Error::decode(format!("unsupported numpy dtype {other:?}"))),
    };
    Ok((order, element))
}

/// Read a map key that may have been packed as a string or as binary.
fn key_str(key: &Value) -> Option<&str> {
    match key {
        Value::String(text) => text.as_str(),
        Value::Binary(bytes) => std::str::from_utf8(bytes).ok(),
        _ => None,
    }
}

fn lookup<'a>(entries: &'a [(Value, Value)], name: &str) -> Option<&'a Value> {
    entries
        .iter()
        .find(|(key, _)| key_str(key) == Some(name))
        .map(|(_, value)| value)
}

fn as_bytes(value: &Value) -> Option<&[u8]> {
    match value {
        Value::Binary(bytes) => Some(bytes),
        Value::String(text) => text.as_bytes().into(),
        _ => None,
    }
}

fn as_text(value: &Value) -> Option<&str> {
    match value {
        Value::String(text) => text.as_str(),
        Value::Binary(bytes) => std::str::from_utf8(bytes).ok(),
        _ => None,
    }
}

/// Whether a value looks like a msgpack-numpy array.
pub fn is_array(value: &Value) -> bool {
    matches!(value, Value::Map(entries) if lookup(entries, "nd").is_some_and(|nd| nd.as_bool() == Some(true)))
}

/// Decode a msgpack-numpy array.
pub fn decode(value: &Value) -> Result<RawArray> {
    let Value::Map(entries) = value else {
        return Err(Error::decode("expected a msgpack map for a numpy array"));
    };
    if lookup(entries, "nd").and_then(Value::as_bool) != Some(true) {
        return Err(Error::decode(
            "msgpack map is not a numpy array (missing nd: true)",
        ));
    }

    let descr = lookup(entries, "type")
        .and_then(as_text)
        .ok_or_else(|| Error::decode("numpy array is missing its dtype descriptor"))?;
    let (order, element) = parse_dtype(descr)?;

    let shape: Vec<usize> = match lookup(entries, "shape") {
        Some(Value::Array(dims)) => dims
            .iter()
            .map(|dim| {
                dim.as_u64().map(|value| value as usize).ok_or_else(|| {
                    Error::decode("numpy array shape contains a non-integer dimension")
                })
            })
            .collect::<Result<_>>()?,
        _ => return Err(Error::decode("numpy array is missing its shape")),
    };

    let data = lookup(entries, "data")
        .and_then(as_bytes)
        .ok_or_else(|| Error::decode("numpy array is missing its data buffer"))?;

    let expected = shape.iter().product::<usize>() * element.width();
    if data.len() != expected {
        return Err(Error::decode(format!(
            "numpy array buffer is {} bytes, but shape {shape:?} of {descr} needs {expected}",
            data.len()
        )));
    }

    Ok(RawArray {
        element,
        shape,
        order,
        data: data.to_vec(),
    })
}

macro_rules! read_elements {
    ($self:expr, $ty:ty, $convert:expr) => {{
        let width = std::mem::size_of::<$ty>();
        $self
            .data
            .chunks_exact(width)
            .map(|chunk| {
                let raw: [u8; std::mem::size_of::<$ty>()] =
                    chunk.try_into().expect("chunks_exact width");
                let value = match $self.order {
                    ByteOrder::Little => <$ty>::from_le_bytes(raw),
                    ByteOrder::Big => <$ty>::from_be_bytes(raw),
                };
                #[allow(clippy::redundant_closure_call)]
                $convert(value)
            })
            .collect()
    }};
}

impl RawArray {
    /// Total element count.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Widen every element to `f32`.
    ///
    /// Integer dtypes are converted, not reinterpreted: a quantized `int8` embedding comes
    /// back as the integer values the server sent.
    pub fn to_f32(&self) -> Vec<f32> {
        match self.element {
            Element::F16 => read_elements!(self, u16, |bits| half::f16::from_bits(bits).to_f32()),
            Element::F32 => read_elements!(self, f32, |value| value),
            Element::F64 => read_elements!(self, f64, |value: f64| value as f32),
            Element::I8 => self
                .data
                .iter()
                .map(|byte| f32::from(*byte as i8))
                .collect(),
            Element::I16 => read_elements!(self, i16, f32::from),
            Element::I32 => read_elements!(self, i32, |value: i32| value as f32),
            Element::I64 => read_elements!(self, i64, |value: i64| value as f32),
            Element::U8 => self.data.iter().map(|byte| f32::from(*byte)).collect(),
            Element::U16 => read_elements!(self, u16, f32::from),
            Element::U32 => read_elements!(self, u32, |value: u32| value as f32),
            Element::U64 => read_elements!(self, u64, |value: u64| value as f32),
        }
    }

    /// Read every element as `f16`, preserving the wire precision when the server sent f16.
    pub fn to_f16(&self) -> Vec<half::f16> {
        match self.element {
            Element::F16 => read_elements!(self, u16, half::f16::from_bits),
            _ => self.to_f32().into_iter().map(half::f16::from_f32).collect(),
        }
    }

    /// Read every element as `u32`, for sparse indices.
    pub fn to_u32(&self) -> Result<Vec<u32>> {
        let values: Vec<i64> = match self.element {
            Element::I8 => self
                .data
                .iter()
                .map(|byte| i64::from(*byte as i8))
                .collect(),
            Element::I16 => read_elements!(self, i16, i64::from),
            Element::I32 => read_elements!(self, i32, i64::from),
            Element::I64 => read_elements!(self, i64, |value| value),
            Element::U8 => self.data.iter().map(|byte| i64::from(*byte)).collect(),
            Element::U16 => read_elements!(self, u16, i64::from),
            Element::U32 => read_elements!(self, u32, i64::from),
            Element::U64 => read_elements!(self, u64, |value: u64| value as i64),
            Element::F16 | Element::F32 | Element::F64 => {
                return Err(Error::decode(
                    "sparse indices arrived as a floating-point array",
                ));
            }
        };
        values
            .into_iter()
            .map(|value| {
                u32::try_from(value)
                    .map_err(|_| Error::decode(format!("sparse index {value} is out of range")))
            })
            .collect()
    }

    /// Rows of a 2-D array, as `f32`.
    pub fn rows_f32(&self) -> Result<Vec<Vec<f32>>> {
        let cols = self.cols()?;
        Ok(self.to_f32().chunks(cols).map(<[f32]>::to_vec).collect())
    }

    /// Rows of a 2-D array, preserving `f16` precision.
    pub fn rows_f16(&self) -> Result<Vec<Vec<half::f16>>> {
        let cols = self.cols()?;
        Ok(self
            .to_f16()
            .chunks(cols)
            .map(<[half::f16]>::to_vec)
            .collect())
    }

    fn cols(&self) -> Result<usize> {
        match self.shape.as_slice() {
            [_, cols] if *cols > 0 => Ok(*cols),
            [_, _] => Ok(1),
            other => Err(Error::decode(format!(
                "expected a 2-D array, got shape {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    /// Build the msgpack-numpy encoding of an `f32` array, as `msgpack_numpy` would.
    pub(crate) fn f32_array(shape: &[usize], values: &[f32]) -> Value {
        raw_array(
            "<f4",
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }

    pub(crate) fn f16_array(shape: &[usize], values: &[f32]) -> Value {
        raw_array(
            "<f2",
            shape,
            values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
                .collect(),
        )
    }

    pub(crate) fn i32_array(shape: &[usize], values: &[i32]) -> Value {
        raw_array(
            "<i4",
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }

    pub(crate) fn raw_array(descr: &str, shape: &[usize], data: Vec<u8>) -> Value {
        Value::Map(vec![
            (Value::Binary(b"nd".to_vec()), Value::Boolean(true)),
            (Value::Binary(b"type".to_vec()), Value::String(descr.into())),
            (Value::Binary(b"kind".to_vec()), Value::Binary(Vec::new())),
            (
                Value::Binary(b"shape".to_vec()),
                Value::Array(shape.iter().map(|dim| Value::from(*dim as u64)).collect()),
            ),
            (Value::Binary(b"data".to_vec()), Value::Binary(data)),
        ])
    }
}

#[cfg(test)]
mod tests {
    // These assertions are about exact values, so exact comparison is the point.
    #![allow(clippy::float_cmp)]

    use super::*;
    use fixtures::*;

    #[test]
    fn decodes_a_dense_f32_vector() {
        let value = f32_array(&[3], &[0.5, -1.5, 2.0]);
        assert!(is_array(&value));
        let array = decode(&value).unwrap();
        assert_eq!(array.element, Element::F32);
        assert_eq!(array.shape, vec![3]);
        assert_eq!(array.to_f32(), vec![0.5, -1.5, 2.0]);
    }

    #[test]
    fn decodes_f16_without_losing_the_wire_precision() {
        let value = f16_array(&[2, 2], &[1.0, 0.25, -3.5, 0.1]);
        let array = decode(&value).unwrap();
        assert_eq!(array.element, Element::F16);
        let rows = array.rows_f16().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][1], half::f16::from_f32(0.25));
        // 0.1 is not representable in f16; the decoder must return the wire bits, not a
        // silently re-rounded value.
        assert_eq!(rows[1][1], half::f16::from_f32(0.1));
        assert_eq!(array.rows_f32().unwrap()[0][0], 1.0);
    }

    #[test]
    fn decodes_sparse_indices() {
        let array = decode(&i32_array(&[4], &[3, 17, 900, 0])).unwrap();
        assert_eq!(array.to_u32().unwrap(), vec![3, 17, 900, 0]);
    }

    #[test]
    fn rejects_negative_sparse_indices() {
        let array = decode(&i32_array(&[2], &[-1, 4])).unwrap();
        assert!(array.to_u32().is_err());
    }

    #[test]
    fn rejects_float_arrays_as_indices() {
        let array = decode(&f32_array(&[2], &[1.0, 2.0])).unwrap();
        assert!(array.to_u32().is_err());
    }

    #[test]
    fn honours_big_endian_buffers() {
        let data: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let array = decode(&raw_array(">f4", &[2], data)).unwrap();
        assert_eq!(array.to_f32(), vec![1.5, -2.0]);
    }

    #[test]
    fn widens_quantized_int8_embeddings() {
        let array = decode(&raw_array("|i1", &[3], vec![0xff, 0x01, 0x80])).unwrap();
        assert_eq!(array.to_f32(), vec![-1.0, 1.0, -128.0]);
    }

    #[test]
    fn rejects_a_buffer_that_does_not_match_its_shape() {
        let truncated = raw_array("<f4", &[4], vec![0; 8]);
        let err = decode(&truncated).unwrap_err();
        assert!(err.to_string().contains("needs 16"), "{err}");
    }

    #[test]
    fn rejects_unsupported_dtypes_and_non_arrays() {
        assert!(decode(&raw_array("<c8", &[1], vec![0; 8])).is_err());
        assert!(decode(&Value::Nil).is_err());
        assert!(!is_array(&Value::Map(vec![(
            Value::String("dense".into()),
            Value::Nil
        )])));
    }

    #[test]
    fn accepts_string_keys_as_well_as_binary_keys() {
        let value = Value::Map(vec![
            (Value::String("nd".into()), Value::Boolean(true)),
            (Value::String("type".into()), Value::String("<f4".into())),
            (
                Value::String("shape".into()),
                Value::Array(vec![Value::from(2u64)]),
            ),
            (
                Value::String("data".into()),
                Value::Binary([1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect()),
            ),
        ]);
        assert_eq!(decode(&value).unwrap().to_f32(), vec![1.0, 2.0]);
    }

    #[test]
    fn round_trips_through_real_msgpack_bytes() {
        let packed = rmp_serde::to_vec(&f32_array(&[2, 2], &[1.0, 2.0, 3.0, 4.0])).unwrap();
        let value: Value = rmp_serde::from_slice(&packed).unwrap();
        let array = decode(&value).unwrap();
        assert_eq!(
            array.rows_f32().unwrap(),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );
    }
}
