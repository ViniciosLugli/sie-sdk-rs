//! Server-sent event framing.
//!
//! The gateway emits exactly one single-line `data: <json>` per chunk, followed by a
//! literal `data: [DONE]`. Multi-line `data:` continuations and the `event:` / `id:` /
//! `retry:` fields are outside that contract and are deliberately not handled.

/// Sentinel that terminates a stream. It is never yielded to the caller.
pub(crate) const DONE: &str = "[DONE]";

/// The payload carried by one SSE line, or `None` for a line that carries none.
///
/// Blank lines are event separators and `:` lines are keep-alive comments.
pub(crate) fn data_payload(line: &str) -> Option<&str> {
    if line.is_empty() || line.starts_with(':') {
        return None;
    }
    let value = line.strip_prefix("data:")?;
    // The SSE spec strips exactly one leading space, not all whitespace.
    Some(value.strip_prefix(' ').unwrap_or(value))
}

/// The error an SSE chunk carries, when it is an error rather than a payload.
///
/// The gateway reports mid-stream failures as an ordinary event whose body is
/// `{"error": {"code": ..., "message": ...}}`.
pub(crate) fn chunk_error(chunk: &serde_json::Value) -> Option<(String, String)> {
    let error = chunk.get("error")?.as_object()?;
    let code = error
        .get("code")
        .and_then(serde_json::Value::as_str)
        .filter(|code| !code.is_empty())
        .unwrap_or("error");
    let message = error
        .get("message")
        .and_then(serde_json::Value::as_str)
        .filter(|message| !message.is_empty())
        .unwrap_or("stream error");
    Some((code.to_string(), message.to_string()))
}

/// Incremental line framing over a byte stream.
#[derive(Debug, Default)]
pub(crate) struct LineDecoder {
    buffer: Vec<u8>,
}

impl LineDecoder {
    /// Feed a chunk and drain every complete line it produced.
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut lines = Vec::new();
        while let Some(index) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.buffer.drain(..=index).collect();
            let trimmed = line.strip_suffix(b"\n").unwrap_or(&line);
            let trimmed = trimmed.strip_suffix(b"\r").unwrap_or(trimmed);
            lines.push(String::from_utf8_lossy(trimmed).into_owned());
        }
        lines
    }

    /// Any trailing bytes left when the connection closed without a final newline.
    pub(crate) fn finish(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        let line = std::mem::take(&mut self.buffer);
        let trimmed = line.strip_suffix(b"\r").unwrap_or(&line);
        Some(String::from_utf8_lossy(trimmed).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_only_data_lines() {
        assert_eq!(data_payload("data: {\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(data_payload("data:{\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(data_payload("data:  spaced"), Some(" spaced"));
        assert_eq!(data_payload(""), None);
        assert_eq!(data_payload(": keep-alive"), None);
        assert_eq!(data_payload("event: message"), None);
        assert_eq!(data_payload("id: 7"), None);
    }

    #[test]
    fn recognises_the_done_sentinel() {
        assert_eq!(data_payload("data: [DONE]"), Some(DONE));
    }

    #[test]
    fn reads_the_error_shape_and_defaults_its_halves() {
        use serde_json::json;

        assert_eq!(
            chunk_error(&json!({"error": {"code": "RESOURCE_EXHAUSTED", "message": "no memory"}})),
            Some(("RESOURCE_EXHAUSTED".to_string(), "no memory".to_string()))
        );
        assert_eq!(
            chunk_error(&json!({"error": {}})),
            Some(("error".to_string(), "stream error".to_string()))
        );
        assert!(chunk_error(&json!({"choices": []})).is_none());
        // A bare string is not the documented error shape.
        assert!(chunk_error(&json!({"error": "boom"})).is_none());
    }

    #[test]
    fn frames_lines_across_chunk_boundaries() {
        let mut decoder = LineDecoder::default();
        assert!(decoder.push(b"data: {\"a\"").is_empty());
        assert_eq!(
            decoder.push(b":1}\n\ndata: [DO"),
            vec!["data: {\"a\":1}", ""]
        );
        assert_eq!(decoder.push(b"NE]\n"), vec!["data: [DONE]"]);
        assert!(decoder.finish().is_none());
    }

    #[test]
    fn strips_carriage_returns_and_keeps_a_trailing_partial_line() {
        let mut decoder = LineDecoder::default();
        assert_eq!(decoder.push(b"data: one\r\ndata: two"), vec!["data: one"]);
        assert_eq!(decoder.finish(), Some("data: two".to_string()));
    }
}
