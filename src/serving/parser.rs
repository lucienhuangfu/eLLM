//! Incremental streaming parser for OpenAI-compatible chat completion output.
//!
//! The parser consumes only newly generated text deltas. It keeps a small
//! internal buffer for tag boundaries and never reparses historical output.
//! Model-specific rules are selected at load time, not inferred from templates.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::{Stream, StreamExt};

use crate::transformer::config::ModelFamily;

pub type RequestId = String;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextDelta {
    pub request_id: RequestId,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub fragment: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    Tagged,
    PrefixedJson,
    RawJson,
    MiniMaxM2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParserEvent {
    Content(String),
    Reasoning(String),
    ToolCall(ToolCall),
    ToolCallDelta(ToolCallDelta),
    Finish,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParserRule {
    pub tool_start: &'static str,
    pub tool_end: &'static str,
    pub think_start: &'static str,
    pub think_end: &'static str,
    pub tool_format: ToolCallFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParserOptions {
    pub rule: ParserRule,
    pub reasoning_parser: bool,
    pub tool_call_parser: bool,
}

impl ParserOptions {
    pub fn new(rule: ParserRule) -> Self {
        Self {
            rule,
            reasoning_parser: true,
            tool_call_parser: true,
        }
    }
}

impl ParserRule {
    pub const fn new(
        tool_start: &'static str,
        tool_end: &'static str,
        think_start: &'static str,
        think_end: &'static str,
        tool_format: ToolCallFormat,
    ) -> Self {
        Self {
            tool_start,
            tool_end,
            think_start,
            think_end,
            tool_format,
        }
    }

    pub const fn qwen() -> Self {
        Self::new(
            "<tool_call>",
            "</tool_call>",
            "<think>",
            "</think>",
            ToolCallFormat::Tagged,
        )
    }

    pub const fn deepseek() -> Self {
        Self::new(
            "<tool_call>",
            "</tool_call>",
            "<think>",
            "</think>",
            ToolCallFormat::Tagged,
        )
    }

    pub const fn hermes() -> Self {
        Self::new(
            "<tool_call>",
            "</tool_call>",
            "<think>",
            "</think>",
            ToolCallFormat::Tagged,
        )
    }

    pub const fn llama3_json() -> Self {
        Self::new(
            "<|python_tag|>",
            "",
            "<think>",
            "</think>",
            ToolCallFormat::RawJson,
        )
    }

    pub const fn mistral() -> Self {
        Self::new(
            "[TOOL_CALLS]",
            "",
            "<think>",
            "</think>",
            ToolCallFormat::PrefixedJson,
        )
    }

    pub const fn minimax_m1() -> Self {
        Self::new(
            "<tool_calls>",
            "</tool_calls>",
            "<think>",
            "</think>",
            ToolCallFormat::Tagged,
        )
    }

    pub const fn minimax_m2() -> Self {
        Self::new(
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            "<think>",
            "</think>",
            ToolCallFormat::MiniMaxM2,
        )
    }

    pub fn for_model_family(family: &ModelFamily) -> Self {
        match family {
            ModelFamily::Qwen => Self::hermes(),
            ModelFamily::Llama => Self::llama3_json(),
            ModelFamily::Mixtral => Self::mistral(),
            ModelFamily::MiniMax => Self::minimax_m1(),
            ModelFamily::MiniMaxM2 => Self::minimax_m1(),
            ModelFamily::Unknown(_) => Self::hermes(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserState {
    Normal,
    Reasoning,
    ToolCall,
    ToolJson,
}

#[derive(Debug, Clone)]
pub struct ParserContext {
    pub state: ParserState,
    pub buffer: String,
    pub tool_json_buffer: String,
    pub options: ParserOptions,
}

impl ParserContext {
    pub fn new(options: ParserOptions) -> Self {
        Self {
            state: ParserState::Normal,
            buffer: String::new(),
            tool_json_buffer: String::new(),
            options,
        }
    }
}

pub trait StreamingParser {
    fn feed(&mut self, delta: &str) -> Vec<ParserEvent>;
}

#[derive(Debug, Clone)]
pub struct IncrementalStreamingParser {
    context: ParserContext,
}

impl IncrementalStreamingParser {
    pub fn new(rule: ParserRule) -> Self {
        Self::with_options(ParserOptions::new(rule))
    }

    pub fn with_options(options: ParserOptions) -> Self {
        Self {
            context: ParserContext::new(options),
        }
    }

    pub fn state(&self) -> ParserState {
        self.context.state
    }
}

impl StreamingParser for IncrementalStreamingParser {
    fn feed(&mut self, delta: &str) -> Vec<ParserEvent> {
        self.context.buffer.push_str(delta);

        let mut events = Vec::new();
        loop {
            match self.context.state {
                ParserState::Normal => {
                    if self.context.buffer.is_empty() {
                        break;
                    }

                    if let Some((marker_kind, marker_pos, marker_len)) =
                        find_next_marker(&self.context.buffer, self.context.options)
                    {
                        if marker_pos > 0 {
                            let content: String = self.context.buffer.drain(..marker_pos).collect();
                            if !content.is_empty() {
                                events.push(ParserEvent::Content(content));
                            }
                        }

                        self.context.buffer.drain(..marker_len);
                        self.context.state = match marker_kind {
                            MarkerKind::ThinkStart => ParserState::Reasoning,
                            MarkerKind::ToolStart => ParserState::ToolCall,
                        };
                        continue;
                    }

                    let keep =
                        longest_suffix_prefix_len(&self.context.buffer, self.context.options);
                    let emit_len = self.context.buffer.len().saturating_sub(keep);
                    if emit_len > 0 {
                        let content: String = self.context.buffer.drain(..emit_len).collect();
                        if !content.is_empty() {
                            events.push(ParserEvent::Content(content));
                        }
                        continue;
                    }

                    break;
                }
                ParserState::Reasoning => {
                    if self.context.buffer.is_empty() {
                        break;
                    }

                    if let Some(end_pos) = self
                        .context
                        .buffer
                        .find(self.context.options.rule.think_end)
                    {
                        if end_pos > 0 {
                            let reasoning: String = self.context.buffer.drain(..end_pos).collect();
                            if !reasoning.is_empty() {
                                events.push(ParserEvent::Reasoning(reasoning));
                            }
                        }

                        self.context
                            .buffer
                            .drain(..self.context.options.rule.think_end.len());
                        self.context.state = ParserState::Normal;
                        continue;
                    }

                    let keep =
                        longest_suffix_prefix_len(&self.context.buffer, self.context.options);
                    let emit_len = self.context.buffer.len().saturating_sub(keep);
                    if emit_len > 0 {
                        let reasoning: String = self.context.buffer.drain(..emit_len).collect();
                        if !reasoning.is_empty() {
                            events.push(ParserEvent::Reasoning(reasoning));
                        }
                        continue;
                    }

                    break;
                }
                ParserState::ToolCall => {
                    self.context.state = ParserState::ToolJson;
                    continue;
                }
                ParserState::ToolJson => {
                    if self.context.buffer.is_empty() {
                        break;
                    }

                    match self.context.options.rule.tool_format {
                        ToolCallFormat::Tagged | ToolCallFormat::MiniMaxM2 => {
                            if let Some(end_pos) =
                                self.context.buffer.find(self.context.options.rule.tool_end)
                            {
                                if end_pos > 0 {
                                    let fragment: String =
                                        self.context.buffer.drain(..end_pos).collect();
                                    if !fragment.is_empty() {
                                        self.context.tool_json_buffer.push_str(&fragment);
                                        events.push(ParserEvent::ToolCallDelta(ToolCallDelta {
                                            fragment,
                                        }));
                                    }
                                }

                                self.context
                                    .buffer
                                    .drain(..self.context.options.rule.tool_end.len());
                                let tool_json = std::mem::take(&mut self.context.tool_json_buffer);

                                if let Some(tool_call) = parse_tool_call_payload(
                                    &tool_json,
                                    self.context.options.rule.tool_format,
                                ) {
                                    events.push(ParserEvent::ToolCall(tool_call));
                                }

                                self.context.state = ParserState::Normal;
                                continue;
                            }

                            let keep = longest_suffix_prefix_len(
                                &self.context.buffer,
                                self.context.options,
                            );
                            let emit_len = self.context.buffer.len().saturating_sub(keep);
                            if emit_len > 0 {
                                let fragment: String =
                                    self.context.buffer.drain(..emit_len).collect();
                                if !fragment.is_empty() {
                                    self.context.tool_json_buffer.push_str(&fragment);
                                    events.push(ParserEvent::ToolCallDelta(ToolCallDelta {
                                        fragment,
                                    }));
                                }
                                continue;
                            }

                            break;
                        }
                        ToolCallFormat::RawJson | ToolCallFormat::PrefixedJson => {
                            let previous_len = self.context.tool_json_buffer.len();
                            let fragment = std::mem::take(&mut self.context.buffer);
                            if !fragment.is_empty() {
                                self.context.tool_json_buffer.push_str(&fragment);
                            }

                            if let Some((tool_calls, consumed)) = parse_complete_tool_call_payload(
                                &self.context.tool_json_buffer,
                                self.context.options.rule.tool_format,
                            ) {
                                let payload_from_fragment =
                                    consumed.saturating_sub(previous_len).min(fragment.len());
                                if payload_from_fragment > 0 {
                                    events.push(ParserEvent::ToolCallDelta(ToolCallDelta {
                                        fragment: fragment[..payload_from_fragment].to_string(),
                                    }));
                                }

                                let remainder = self.context.tool_json_buffer.split_off(consumed);
                                self.context.tool_json_buffer.clear();
                                if !remainder.is_empty() {
                                    self.context.buffer = remainder;
                                }
                                for tool_call in tool_calls {
                                    events.push(ParserEvent::ToolCall(tool_call));
                                }
                                self.context.state = ParserState::Normal;
                                continue;
                            }

                            if !fragment.is_empty() {
                                events.push(ParserEvent::ToolCallDelta(ToolCallDelta { fragment }));
                            }

                            break;
                        }
                    }
                }
            }
        }

        events
    }
}

fn parse_tool_call_payload(text: &str, format: ToolCallFormat) -> Option<ToolCall> {
    match format {
        ToolCallFormat::Tagged => {
            let value: Value = serde_json::from_str(text).ok()?;
            extract_tool_call_value(&value)?.into_iter().next()
        }
        ToolCallFormat::PrefixedJson => {
            let (tool_calls, _) = extract_complete_prefixed_tool_call(text)?;
            tool_calls.into_iter().next()
        }
        ToolCallFormat::RawJson => {
            let (start, end) = extract_complete_json_prefix_range(text)?;
            let value: Value = serde_json::from_str(&text[start..end]).ok()?;
            extract_tool_call_value(&value)?.into_iter().next()
        }
        ToolCallFormat::MiniMaxM2 => parse_minimax_m2_tool_call(text)?.into_iter().next(),
    }
}

fn parse_complete_tool_call_payload(
    text: &str,
    format: ToolCallFormat,
) -> Option<(Vec<ToolCall>, usize)> {
    match format {
        ToolCallFormat::RawJson => {
            let (start, end) = extract_complete_json_prefix_range(text)?;
            let value: Value = serde_json::from_str(&text[start..end]).ok()?;
            Some((extract_tool_call_value(&value)?, end))
        }
        ToolCallFormat::PrefixedJson => {
            let (tool_calls, consumed) = extract_complete_prefixed_tool_call(text)?;
            Some((tool_calls, consumed))
        }
        _ => None,
    }
}

fn parse_minimax_m2_tool_call(text: &str) -> Option<Vec<ToolCall>> {
    let invoke_start = text.find("<invoke")?;
    let invoke_end = text[invoke_start..].find("</invoke>")? + invoke_start;
    let invoke_block = &text[invoke_start..invoke_end];
    let extract_attr = |source: &str, attr: &str| -> Option<String> {
        let pattern = format!(r#"{attr}=""#);
        let start = source.find(&pattern)? + pattern.len();
        let end = source[start..].find('"')? + start;
        Some(source[start..end].to_string())
    };
    let name = extract_attr(invoke_block, "name")?;

    let mut arguments = serde_json::Map::new();
    let mut search_offset = 0usize;

    while let Some(parameter_start_rel) = invoke_block[search_offset..].find("<parameter") {
        let parameter_start = search_offset + parameter_start_rel;
        let parameter_tag_end = invoke_block[parameter_start..].find('>')? + parameter_start;
        let parameter_tag = &invoke_block[parameter_start..=parameter_tag_end];
        let parameter_name = extract_attr(parameter_tag, "name")?;
        let parameter_close =
            invoke_block[parameter_tag_end + 1..].find("</parameter>")? + parameter_tag_end + 1;
        let parameter_value = invoke_block[parameter_tag_end + 1..parameter_close].trim();
        let value = serde_json::from_str(parameter_value)
            .unwrap_or_else(|_| Value::String(parameter_value.to_string()));
        arguments.insert(parameter_name.to_string(), value);
        search_offset = parameter_close + "</parameter>".len();
    }

    Some(vec![ToolCall {
        name,
        arguments: Value::Object(arguments),
    }])
}

fn extract_complete_json_prefix_range(text: &str) -> Option<(usize, usize)> {
    let start = text
        .char_indices()
        .find(|(_, ch)| !ch.is_whitespace())
        .map(|(index, _)| index)?;
    let mut chars = text[start..].char_indices();
    let (_, first) = chars.next()?;
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escape = false;

    match first {
        '{' => stack.push('}'),
        '[' => stack.push(']'),
        _ => return None,
    }

    for (offset, ch) in chars {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            match ch {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                if stack.pop() != Some(ch) {
                    return None;
                }
                if stack.is_empty() {
                    return Some((start, start + offset + ch.len_utf8()));
                }
            }
            _ => {}
        }
    }

    None
}

fn extract_complete_prefixed_tool_call(text: &str) -> Option<(Vec<ToolCall>, usize)> {
    let json_start = text.find('{')?;
    let name = text[..json_start].trim().to_string();
    if name.is_empty() {
        return None;
    }

    let (start, end) = extract_complete_json_prefix_range(&text[json_start..])?;
    let arguments: Value =
        serde_json::from_str(&text[json_start + start..json_start + end]).ok()?;
    Some((vec![ToolCall { name, arguments }], json_start + end))
}

fn extract_tool_call_value(value: &Value) -> Option<Vec<ToolCall>> {
    if let Some(array) = value.as_array() {
        let mut tool_calls = Vec::with_capacity(array.len());
        for item in array {
            let call = extract_tool_call_value(item)?;
            tool_calls.extend(call);
        }
        return Some(tool_calls);
    }

    let object = value.as_object()?;

    if let Some(function) = object.get("function").and_then(Value::as_object) {
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .or_else(|| object.get("name").and_then(Value::as_str))?
            .to_string();
        let arguments = function
            .get("arguments")
            .cloned()
            .or_else(|| object.get("arguments").cloned())
            .unwrap_or(Value::Null);
        return Some(vec![ToolCall { name, arguments }]);
    }

    let name = object.get("name").and_then(Value::as_str)?.to_string();
    let arguments = object.get("arguments").cloned().unwrap_or(Value::Null);

    Some(vec![ToolCall { name, arguments }])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkerKind {
    ThinkStart,
    ToolStart,
}

fn find_next_marker(text: &str, options: ParserOptions) -> Option<(MarkerKind, usize, usize)> {
    let think_start = if options.reasoning_parser && !options.rule.think_start.is_empty() {
        text.find(options.rule.think_start)
    } else {
        None
    };
    let tool_start = if options.tool_call_parser && !options.rule.tool_start.is_empty() {
        text.find(options.rule.tool_start)
    } else {
        None
    };

    match (think_start, tool_start) {
        (Some(think_pos), Some(tool_pos)) if think_pos <= tool_pos => Some((
            MarkerKind::ThinkStart,
            think_pos,
            options.rule.think_start.len(),
        )),
        (Some(_think_pos), Some(tool_pos)) => Some((
            MarkerKind::ToolStart,
            tool_pos,
            options.rule.tool_start.len(),
        )),
        (Some(think_pos), None) => Some((
            MarkerKind::ThinkStart,
            think_pos,
            options.rule.think_start.len(),
        )),
        (None, Some(tool_pos)) => Some((
            MarkerKind::ToolStart,
            tool_pos,
            options.rule.tool_start.len(),
        )),
        (None, None) => None,
    }
}

fn longest_suffix_prefix_len(text: &str, options: ParserOptions) -> usize {
    let bytes = text.as_bytes();
    let mut keep = 0usize;

    let markers = [
        options
            .reasoning_parser
            .then_some(options.rule.think_start)
            .filter(|marker| !marker.is_empty()),
        options
            .tool_call_parser
            .then_some(options.rule.tool_start)
            .filter(|marker| !marker.is_empty()),
    ];

    for marker in markers.into_iter().flatten() {
        let marker_bytes = marker.as_bytes();
        let limit = bytes.len().min(marker_bytes.len());

        for suffix_len in 1..=limit {
            if bytes[bytes.len() - suffix_len..] == marker_bytes[..suffix_len] {
                keep = keep.max(suffix_len);
            }
        }
    }

    keep
}

pub fn parser_event_stream<S, P>(mut parser: P, mut deltas: S) -> impl Stream<Item = ParserEvent>
where
    S: Stream<Item = TextDelta> + Unpin,
    P: StreamingParser + Unpin,
{
    async_stream::stream! {
        while let Some(delta) = deltas.next().await {
            for event in parser.feed(&delta.text) {
                yield event;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_events(parser: &mut impl StreamingParser, chunks: &[&str]) -> Vec<ParserEvent> {
        let mut events = Vec::new();

        for chunk in chunks {
            events.extend(parser.feed(chunk));
        }

        events
    }

    #[test]
    fn streams_reasoning_incrementally() {
        let mut parser = IncrementalStreamingParser::new(ParserRule::qwen());
        let events = collect_events(
            &mut parser,
            &["hello ", "<think>an", "alyzing", "</think> world"],
        );

        assert_eq!(
            events,
            vec![
                ParserEvent::Content("hello ".to_string()),
                ParserEvent::Reasoning("an".to_string()),
                ParserEvent::Reasoning("alyzing".to_string()),
                ParserEvent::Content(" world".to_string()),
            ]
        );
    }

    #[test]
    fn streams_tool_call_across_boundaries() {
        let mut parser = IncrementalStreamingParser::new(ParserRule::qwen());
        let events = collect_events(
            &mut parser,
            &[
                "text ",
                "<tool_call>{\"name\":\"search\",\"arguments\":",
                "{\"query\":\"Tesla\"}",
                "}</tool_call> done",
            ],
        );

        assert_eq!(
            events,
            vec![
                ParserEvent::Content("text ".to_string()),
                ParserEvent::ToolCallDelta(ToolCallDelta {
                    fragment: "{\"name\":\"search\",\"arguments\":".to_string()
                }),
                ParserEvent::ToolCallDelta(ToolCallDelta {
                    fragment: "{\"query\":\"Tesla\"}".to_string()
                }),
                ParserEvent::ToolCallDelta(ToolCallDelta {
                    fragment: "}".to_string()
                }),
                ParserEvent::ToolCall(ToolCall {
                    name: "search".to_string(),
                    arguments: serde_json::json!({"query":"Tesla"}),
                }),
                ParserEvent::Content(" done".to_string()),
            ]
        );
    }

    #[test]
    fn keeps_partial_tags_in_buffer() {
        let mut parser = IncrementalStreamingParser::new(ParserRule::qwen());
        let events = parser.feed("abc<thi");
        assert_eq!(events, vec![ParserEvent::Content("abc".to_string())]);
        assert_eq!(parser.state(), ParserState::Normal);

        let events = parser.feed("nk>hello</think>");
        assert_eq!(events, vec![ParserEvent::Reasoning("hello".to_string())]);
    }

    #[test]
    fn parses_nested_function_tool_call_shape() {
        let text = r#"{"function":{"name":"search","arguments":{"query":"mars"}}}"#;
        let tool_call =
            parse_tool_call_payload(text, ToolCallFormat::Tagged).expect("tool call should parse");

        assert_eq!(tool_call.name, "search");
        assert_eq!(tool_call.arguments, serde_json::json!({"query": "mars"}));
    }

    #[test]
    fn parses_llama3_json_tool_call_across_boundaries() {
        let mut parser = IncrementalStreamingParser::new(ParserRule::llama3_json());
        let events = collect_events(
            &mut parser,
            &[
                "intro ",
                "<|python_tag|>{\"name\":\"search\",\"arguments\":",
                "{\"query\":\"Mars\"}} tail",
            ],
        );

        assert_eq!(
            events,
            vec![
                ParserEvent::Content("intro ".to_string()),
                ParserEvent::ToolCallDelta(ToolCallDelta {
                    fragment: "{\"name\":\"search\",\"arguments\":".to_string()
                }),
                ParserEvent::ToolCallDelta(ToolCallDelta {
                    fragment: "{\"query\":\"Mars\"}}".to_string()
                }),
                ParserEvent::ToolCall(ToolCall {
                    name: "search".to_string(),
                    arguments: serde_json::json!({"query":"Mars"}),
                }),
                ParserEvent::Content(" tail".to_string()),
            ]
        );
    }

    #[test]
    fn parses_minimax_m1_tool_call() {
        let mut parser = IncrementalStreamingParser::new(ParserRule::minimax_m1());
        let events = collect_events(
            &mut parser,
            &[
                "before ",
                "<tool_calls>{\"name\":\"search\",\"arguments\":",
                "{\"query\":\"saturn\"}}</tool_calls> after",
            ],
        );

        assert_eq!(
            events,
            vec![
                ParserEvent::Content("before ".to_string()),
                ParserEvent::ToolCallDelta(ToolCallDelta {
                    fragment: "{\"name\":\"search\",\"arguments\":".to_string()
                }),
                ParserEvent::ToolCallDelta(ToolCallDelta {
                    fragment: "{\"query\":\"saturn\"}}".to_string()
                }),
                ParserEvent::ToolCall(ToolCall {
                    name: "search".to_string(),
                    arguments: serde_json::json!({"query":"saturn"}),
                }),
                ParserEvent::Content(" after".to_string()),
            ]
        );
    }

    #[test]
    fn parses_minimax_m2_tool_call() {
        let mut parser = IncrementalStreamingParser::new(ParserRule::minimax_m2());
        let events = collect_events(
            &mut parser,
            &[
                "before ",
                "<minimax:tool_call><invoke name=\"search\"><parameter name=\"query\">",
                "\"venus\"</parameter></invoke></minimax:tool_call> after",
            ],
        );

        assert_eq!(
            events,
            vec![
                ParserEvent::Content("before ".to_string()),
                ParserEvent::ToolCallDelta(ToolCallDelta {
                    fragment: "<invoke name=\"search\"><parameter name=\"query\">".to_string()
                }),
                ParserEvent::ToolCallDelta(ToolCallDelta {
                    fragment: "\"venus\"</parameter></invoke>".to_string()
                }),
                ParserEvent::ToolCall(ToolCall {
                    name: "search".to_string(),
                    arguments: serde_json::json!({"query":"venus"}),
                }),
                ParserEvent::Content(" after".to_string()),
            ]
        );
    }

    #[test]
    fn ignores_invalid_tool_json_until_complete_parse() {
        let mut parser = IncrementalStreamingParser::new(ParserRule::qwen());
        let events = parser.feed("<tool_call>{\"name\":\"search\",\"arguments\":");

        assert_eq!(
            events,
            vec![ParserEvent::ToolCallDelta(ToolCallDelta {
                fragment: "{\"name\":\"search\",\"arguments\":".to_string()
            })]
        );
        assert_eq!(parser.state(), ParserState::ToolJson);
    }

    #[test]
    fn passes_through_reasoning_tags_when_reasoning_parser_is_disabled() {
        let mut parser = IncrementalStreamingParser::with_options(ParserOptions {
            rule: ParserRule::qwen(),
            reasoning_parser: false,
            tool_call_parser: true,
        });

        let events = collect_events(&mut parser, &["hello ", "<think>keep", "</think> world"]);

        assert_eq!(
            events,
            vec![
                ParserEvent::Content("hello ".to_string()),
                ParserEvent::Content("<think>keep".to_string()),
                ParserEvent::Content("</think> world".to_string()),
            ]
        );
    }

    #[test]
    fn passes_through_tool_tags_when_tool_call_parser_is_disabled() {
        let mut parser = IncrementalStreamingParser::with_options(ParserOptions {
            rule: ParserRule::qwen(),
            reasoning_parser: true,
            tool_call_parser: false,
        });

        let events = collect_events(
            &mut parser,
            &[
                "text ",
                "<tool_call>{\"name\":\"search\",\"arguments\":",
                "{\"query\":\"Tesla\"}}",
                "</tool_call> done",
            ],
        );

        assert_eq!(
            events,
            vec![
                ParserEvent::Content("text ".to_string()),
                ParserEvent::Content("<tool_call>{\"name\":\"search\",\"arguments\":".to_string()),
                ParserEvent::Content("{\"query\":\"Tesla\"}}".to_string()),
                ParserEvent::Content("</tool_call> done".to_string()),
            ]
        );
    }
}
