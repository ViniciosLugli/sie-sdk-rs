//! Small accessors over decoded msgpack values.
//!
//! Response bodies are walked rather than deserialized because tensors are carried in the
//! `msgpack-numpy` encoding, which no `serde` impl can express.

use rmpv::Value as MsgValue;
use serde_json::{Map, Number, Value};

/// Look up a map key, accepting the string or binary encoding of the key.
pub(crate) fn get<'a>(value: &'a MsgValue, key: &str) -> Option<&'a MsgValue> {
    let MsgValue::Map(entries) = value else {
        return None;
    };
    entries
        .iter()
        .find(|(name, _)| key_str(name) == Some(key))
        .map(|(_, value)| value)
        .filter(|value| !matches!(value, MsgValue::Nil))
}

fn key_str(key: &MsgValue) -> Option<&str> {
    match key {
        MsgValue::String(text) => text.as_str(),
        MsgValue::Binary(bytes) => std::str::from_utf8(bytes).ok(),
        _ => None,
    }
}

/// Read a value as text, accepting the string or binary encoding.
pub(crate) fn text(value: &MsgValue) -> Option<&str> {
    match value {
        MsgValue::String(text) => text.as_str(),
        MsgValue::Binary(bytes) => std::str::from_utf8(bytes).ok(),
        _ => None,
    }
}

/// Read a map key as text.
pub(crate) fn get_text<'a>(value: &'a MsgValue, key: &str) -> Option<&'a str> {
    text(get(value, key)?)
}

/// Read a map key as an owned string.
pub(crate) fn get_string(value: &MsgValue, key: &str) -> Option<String> {
    get_text(value, key).map(str::to_string)
}

/// Read a map key as a float, accepting any numeric encoding.
pub(crate) fn get_f64(value: &MsgValue, key: &str) -> Option<f64> {
    let value = get(value, key)?;
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|v| v as f64))
        .or_else(|| value.as_u64().map(|v| v as f64))
}

/// Read a map key as an unsigned integer.
pub(crate) fn get_u64(value: &MsgValue, key: &str) -> Option<u64> {
    let value = get(value, key)?;
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|v| u64::try_from(v).ok()))
}

/// Read a map key as a signed integer.
pub(crate) fn get_i64(value: &MsgValue, key: &str) -> Option<i64> {
    let value = get(value, key)?;
    value.as_i64().or_else(|| value.as_u64().map(|v| v as i64))
}

/// Read a map key as an array.
pub(crate) fn get_array<'a>(value: &'a MsgValue, key: &str) -> Option<&'a [MsgValue]> {
    match get(value, key)? {
        MsgValue::Array(items) => Some(items),
        _ => None,
    }
}

/// Translate a decoded msgpack value into JSON, for fields the API leaves opaque.
///
/// Binary payloads have no JSON representation and become base64 strings; every other type
/// maps directly.
pub(crate) fn to_json(value: &MsgValue) -> Value {
    use base64::Engine as _;

    match value {
        MsgValue::Nil => Value::Null,
        MsgValue::Boolean(flag) => Value::Bool(*flag),
        MsgValue::Integer(number) => number
            .as_i64()
            .map(Value::from)
            .or_else(|| number.as_u64().map(Value::from))
            .unwrap_or(Value::Null),
        MsgValue::F32(number) => {
            Number::from_f64(f64::from(*number)).map_or(Value::Null, Value::Number)
        }
        MsgValue::F64(number) => Number::from_f64(*number).map_or(Value::Null, Value::Number),
        MsgValue::String(text) => text
            .as_str()
            .map_or(Value::Null, |text| Value::String(text.to_string())),
        // Binary and extension payloads have no JSON form; base64 keeps them readable.
        MsgValue::Binary(bytes) | MsgValue::Ext(_, bytes) => {
            Value::String(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        MsgValue::Array(items) => Value::Array(items.iter().map(to_json).collect()),
        MsgValue::Map(entries) => Value::Object(
            entries
                .iter()
                .filter_map(|(key, value)| Some((key_str(key)?.to_string(), to_json(value))))
                .collect::<Map<_, _>>(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(entries: &[(&str, MsgValue)]) -> MsgValue {
        MsgValue::Map(
            entries
                .iter()
                .map(|(key, value)| (MsgValue::from(*key), value.clone()))
                .collect(),
        )
    }

    #[test]
    fn reads_keys_in_either_encoding() {
        let value = MsgValue::Map(vec![
            (MsgValue::Binary(b"a".to_vec()), MsgValue::from("one")),
            (MsgValue::from("b"), MsgValue::from(2u64)),
        ]);
        assert_eq!(get_text(&value, "a"), Some("one"));
        assert_eq!(get_u64(&value, "b"), Some(2));
        assert_eq!(get(&value, "missing"), None);
    }

    #[test]
    fn an_explicit_nil_reads_as_absent() {
        let value = map(&[("id", MsgValue::Nil)]);
        assert!(get(&value, "id").is_none());
        assert_eq!(get_string(&value, "id"), None);
    }

    #[test]
    fn numbers_are_read_through_any_encoding() {
        let value = map(&[
            ("int", MsgValue::from(3i64)),
            ("float", MsgValue::F32(1.5)),
            ("double", MsgValue::F64(2.5)),
            ("negative", MsgValue::from(-4i64)),
        ]);
        assert_eq!(get_f64(&value, "int"), Some(3.0));
        assert_eq!(get_f64(&value, "float"), Some(1.5));
        assert_eq!(get_f64(&value, "double"), Some(2.5));
        assert_eq!(get_u64(&value, "negative"), None);
        assert_eq!(get_i64(&value, "negative"), Some(-4));
    }

    #[test]
    fn converts_nested_structures_to_json() {
        let value = map(&[
            ("text", MsgValue::from("hi")),
            ("count", MsgValue::from(2u64)),
            (
                "nested",
                MsgValue::Array(vec![MsgValue::Boolean(true), MsgValue::Nil]),
            ),
        ]);
        assert_eq!(
            to_json(&value),
            json!({"text": "hi", "count": 2, "nested": [true, null]})
        );
    }

    #[test]
    fn binary_payloads_become_base64() {
        assert_eq!(to_json(&MsgValue::Binary(vec![1, 2, 3])), json!("AQID"));
    }
}
