//! Text generation: the native `/v1/generate` shape plus the OpenAI-compatible ones.

// Wire-mirror types: field names are the API contract itself, and the ones whose
// meaning is not obvious carry their own doc comment.
#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::RequestMetadata;

/// Why a native generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    ContentFilter,
    Error,
}

/// Why an OpenAI-compatible completion stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatFinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

/// Who authored a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
    Developer,
}

/// A constraint on the shape of the generated text.
///
/// Exactly one of the three forms applies; the enum makes that structural rather than a
/// runtime check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Grammar {
    /// Output must validate against this JSON schema.
    JsonSchema {
        json_schema: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    /// Output must match this regular expression.
    Regex {
        regex: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    /// Output must parse under this EBNF grammar.
    Ebnf {
        ebnf: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

impl Grammar {
    /// Constrain output to a JSON schema.
    pub fn json_schema(schema: Value) -> Self {
        Self::JsonSchema {
            json_schema: schema,
            label: None,
            strict: None,
        }
    }

    /// Constrain output to a regular expression.
    pub fn regex(pattern: impl Into<String>) -> Self {
        Self::Regex {
            regex: pattern.into(),
            label: None,
            strict: None,
        }
    }

    /// Constrain output to an EBNF grammar.
    pub fn ebnf(grammar: impl Into<String>) -> Self {
        Self::Ebnf {
            ebnf: grammar.into(),
            label: None,
            strict: None,
        }
    }
}

/// Token counts for a native generation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationUsage {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub prompt_tokens: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub completion_tokens: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credits_charged: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_book_version: Option<String>,
}

/// A completed native generation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerateResult {
    pub model: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<GenerationUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_id: Option<String>,
    /// Time to first token, in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    /// Time per output token, in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tpot_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<RequestMetadata>,
}

/// One event of a native generation stream.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerateChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Monotonic within one attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seq: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_delta: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<Value>>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub done: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// Present only on the terminal chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<GenerationUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
}

/// An image referenced by a chat content part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatImageUrl {
    /// A bare URL or `data:` URI.
    Url(String),
    /// The object form `OpenAI` also accepts.
    Object { url: String },
}

/// One part of a multimodal chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ChatImageUrl>,
}

impl ChatContentPart {
    /// A text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            kind: "text".to_string(),
            text: Some(text.into()),
            image_url: None,
        }
    }

    /// An image part, referenced by URL or `data:` URI.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self {
            kind: "image_url".to_string(),
            text: None,
            image_url: Some(ChatImageUrl::Url(url.into())),
        }
    }
}

/// Message content: plain text, or an ordered list of parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

/// One message in a chat conversation. Used both in requests and in responses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
}

impl ChatMessage {
    fn simple(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(ChatContent::Text(content.into())),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// A system prompt.
    pub fn system(content: impl Into<String>) -> Self {
        Self::simple(ChatRole::System, content)
    }

    /// A user turn.
    pub fn user(content: impl Into<String>) -> Self {
        Self::simple(ChatRole::User, content)
    }

    /// An assistant turn.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::simple(ChatRole::Assistant, content)
    }

    /// A developer instruction.
    pub fn developer(content: impl Into<String>) -> Self {
        Self::simple(ChatRole::Developer, content)
    }

    /// The result of a tool call, answering `tool_call_id`.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: Some(tool_call_id.into()),
            ..Self::simple(ChatRole::Tool, content)
        }
    }

    /// A user turn made of ordered parts, for multimodal input.
    pub fn user_parts(parts: impl IntoIterator<Item = ChatContentPart>) -> Self {
        Self {
            role: ChatRole::User,
            content: Some(ChatContent::Parts(parts.into_iter().collect())),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// The message text, when the content is plain text.
    pub fn text(&self) -> Option<&str> {
        match &self.content {
            Some(ChatContent::Text(text)) => Some(text),
            _ => None,
        }
    }
}

/// Token counts for a chat completion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatUsage {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub prompt_tokens: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub completion_tokens: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub total_tokens: u64,
}

/// One completion candidate.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatChoice {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<ChatFinishReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

/// A completed chat completion.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletion {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub id: String,
    #[serde(
        default,
        deserialize_with = "crate::types::null_as_default",
        rename = "object"
    )]
    pub object_kind: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub created: i64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub choices: Vec<ChatChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<RequestMetadata>,
}

impl ChatCompletion {
    /// The first choice's text, which is what single-candidate callers want.
    pub fn text(&self) -> Option<&str> {
        self.choices.first()?.message.as_ref()?.text()
    }
}

/// The incremental part of a streamed choice.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
}

/// One streamed choice.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub index: u32,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub delta: ChatDelta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<ChatFinishReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

/// One event of a chat completion stream.
///
/// The terminal usage-only chunk has an empty `choices` list; it appears only when the
/// request asked for it through `stream_options`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub id: String,
    #[serde(
        default,
        deserialize_with = "crate::types::null_as_default",
        rename = "object"
    )]
    pub object_kind: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub created: i64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub choices: Vec<ChatChunkChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatUsage>,
}

impl ChatCompletionChunk {
    /// The text this chunk adds, if any.
    pub fn delta(&self) -> Option<&str> {
        self.choices.first()?.delta.content.as_deref()
    }
}

/// A message in a Responses-API request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseInputMessage {
    pub role: ChatRole,
    pub content: ChatContent,
}

impl ResponseInputMessage {
    /// A user turn.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: ChatContent::Text(content.into()),
        }
    }

    /// A system prompt.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: ChatContent::Text(content.into()),
        }
    }
}

/// One text block of a response output message.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseOutputText {
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub text: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub annotations: Vec<Value>,
}

/// One output message of a response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseOutputMessage {
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub id: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub role: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub status: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub content: Vec<ResponseOutputText>,
}

/// Token counts for a response.
///
/// The key names differ from [`ChatUsage`]: this is the Responses API's own vocabulary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseUsage {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub input_tokens: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub output_tokens: u64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub total_tokens: u64,
}

/// A completed response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseResult {
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub id: String,
    #[serde(
        default,
        deserialize_with = "crate::types::null_as_default",
        rename = "object"
    )]
    pub object_kind: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub created_at: i64,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub model: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub status: String,
    #[serde(default, deserialize_with = "crate::types::null_as_default")]
    pub output: Vec<ResponseOutputMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<RequestMetadata>,
}

impl ResponseResult {
    /// The first output message's text.
    pub fn text(&self) -> Option<&str> {
        Some(self.output.first()?.content.first()?.text.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn grammar_serializes_to_exactly_one_arm() {
        assert_eq!(
            serde_json::to_value(Grammar::json_schema(json!({"type": "object"}))).unwrap(),
            json!({"json_schema": {"type": "object"}})
        );
        assert_eq!(
            serde_json::to_value(Grammar::regex("[0-9]+")).unwrap(),
            json!({"regex": "[0-9]+"})
        );
        assert_eq!(
            serde_json::to_value(Grammar::ebnf("root ::= \"a\"")).unwrap(),
            json!({"ebnf": "root ::= \"a\""})
        );
    }

    #[test]
    fn grammar_carries_optional_label_and_strict() {
        let grammar = Grammar::Regex {
            regex: "a+".to_string(),
            label: Some("digits".to_string()),
            strict: Some(true),
        };
        assert_eq!(
            serde_json::to_value(grammar).unwrap(),
            json!({"regex": "a+", "label": "digits", "strict": true})
        );
    }

    #[test]
    fn chat_messages_serialize_only_what_was_set() {
        assert_eq!(
            serde_json::to_value(ChatMessage::user("hi")).unwrap(),
            json!({"role": "user", "content": "hi"})
        );
        assert_eq!(
            serde_json::to_value(ChatMessage::tool("call_1", "42")).unwrap(),
            json!({"role": "tool", "content": "42", "tool_call_id": "call_1"})
        );
    }

    #[test]
    fn multimodal_messages_keep_their_part_order() {
        let message = ChatMessage::user_parts([
            ChatContentPart::text("What is this?"),
            ChatContentPart::image_url("https://example.com/a.png"),
        ]);
        assert_eq!(
            serde_json::to_value(message).unwrap(),
            json!({"role": "user", "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": "https://example.com/a.png"}
            ]})
        );
    }

    #[test]
    fn completions_expose_their_first_choice_text() {
        let completion: ChatCompletion = serde_json::from_value(json!({
            "id": "cmpl-1", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        }))
        .unwrap();
        assert_eq!(completion.text(), Some("hello"));
        assert_eq!(completion.object_kind, "chat.completion");
        assert_eq!(
            completion.choices[0].finish_reason,
            Some(ChatFinishReason::Stop)
        );
        assert_eq!(completion.usage.unwrap().total_tokens, 4);
    }

    #[test]
    fn stream_chunks_expose_their_delta_and_tolerate_the_usage_only_tail() {
        let chunk: ChatCompletionChunk = serde_json::from_value(json!({
            "id": "cmpl-1", "choices": [{"index": 0, "delta": {"content": "he"}}]
        }))
        .unwrap();
        assert_eq!(chunk.delta(), Some("he"));

        let tail: ChatCompletionChunk = serde_json::from_value(json!({
            "id": "cmpl-1", "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        }))
        .unwrap();
        assert!(tail.delta().is_none());
        assert_eq!(tail.usage.unwrap().completion_tokens, 1);
    }

    #[test]
    fn responses_expose_their_first_output_text() {
        let result: ResponseResult = serde_json::from_value(json!({
            "id": "resp-1", "object": "response", "model": "m", "status": "completed",
            "output": [{"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "answer", "annotations": []}]}],
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
        }))
        .unwrap();
        assert_eq!(result.text(), Some("answer"));
        assert_eq!(result.usage.unwrap().input_tokens, 2);
    }

    #[test]
    fn native_generate_results_decode() {
        let result: GenerateResult = serde_json::from_value(json!({
            "model": "m", "text": "out", "finish_reason": "length",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3,
                      "credits_charged": 7, "rate_book_version": "2026-01"},
            "ttft_ms": 12.5
        }))
        .unwrap();
        assert_eq!(result.finish_reason, Some(FinishReason::Length));
        assert_eq!(result.usage.unwrap().credits_charged, Some(7));
        assert_eq!(result.ttft_ms, Some(12.5));
    }
}
