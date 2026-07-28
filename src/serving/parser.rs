//! High-performance incremental streaming parser for OpenAI-compatible output.
//!
//! Key optimizations:
//! - Zero-copy events: `ParserEvent<'a>` borrows directly from internal buffer.
//! - Cursor-based buffer: O(1) consumption, no per-token memmove.
//! - Precomputed active markers: no per-call array rebuild.
//! - Fast-path suffix check: single last-byte comparison short-circuits.
//! - JsonScanner tracks consumed bytes: eliminates redundant extract_json_range.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::transformer::config::ModelFamily;

/// Maximum accumulated tool-call buffer before forced recovery (256 KiB).
const MAX_TOOL_BUF: usize = 256 * 1024;
/// Compact the buffer when cursor exceeds this threshold.
const COMPACT_THRESHOLD: usize = 4096;

// ─── Public Types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    Tagged,
    PrefixedJson,
    RawJson,
    MiniMaxM2,
}

/// Zero-copy parser event. `Content`, `Reasoning`, and `ToolCallDelta` borrow
/// from the parser's internal buffer and are valid until the next `feed()` call.
/// `ToolCall` is owned because it is parsed from cross-feed accumulated state.
#[derive(Debug, Clone, PartialEq)]
pub enum ParserEvent<'a> {
    Content(&'a str),
    Reasoning(&'a str),
    ToolCall(ToolCall),
    ToolCallDelta(&'a str),
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

impl Default for ParserOptions {
    fn default() -> Self {
        Self::new(ParserRule::qwen())
    }
}

// ─── ParserRule Presets ──────────────────────────────────────────────────────

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

    /// Standard XML-tagged format (Qwen / DeepSeek / Hermes).
    pub const fn qwen() -> Self {
        Self::new(
            "\x3ctool_call\x3e",
            "\x3c/tool_call\x3e",
            "\x3cthink\x3e",
            "\x3c/think\x3e",
            ToolCallFormat::Tagged,
        )
    }

    pub const fn llama3_json() -> Self {
        Self::new(
            "\x3c|python_tag|\x3e",
            "",
            "\x3cthink\x3e",
            "\x3c/think\x3e",
            ToolCallFormat::RawJson,
        )
    }

    pub const fn mistral() -> Self {
        Self::new(
            "[TOOL_CALLS]",
            "",
            "\x3cthink\x3e",
            "\x3c/think\x3e",
            ToolCallFormat::PrefixedJson,
        )
    }

    pub const fn minimax_m1() -> Self {
        Self::new(
            "\x3ctool_calls\x3e",
            "\x3c/tool_calls\x3e",
            "\x3cthink\x3e",
            "\x3c/think\x3e",
            ToolCallFormat::Tagged,
        )
    }

    pub const fn minimax_m2() -> Self {
        Self::new(
            "\x3cminimax:tool_call\x3e",
            "\x3c/minimax:tool_call\x3e",
            "\x3cthink\x3e",
            "\x3c/think\x3e",
            ToolCallFormat::MiniMaxM2,
        )
    }

    pub fn for_model_family(family: &ModelFamily) -> Self {
        match family {
            ModelFamily::Qwen => Self::qwen(),
            ModelFamily::Llama => Self::llama3_json(),
            ModelFamily::Mixtral => Self::mistral(),
            ModelFamily::MiniMax | ModelFamily::MiniMaxM2 => Self::minimax_m1(),
            ModelFamily::Unknown(_) => Self::qwen(),
        }
    }
}

// ─── Parser State ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserState {
    Normal,
    Reasoning,
    ToolJson,
}

/// Incremental JSON bracket-depth scanner with byte-offset tracking.
#[derive(Debug, Clone)]
struct JsonScanner {
    depth: u32,
    in_string: bool,
    escape: bool,
    started: bool,
    closed: bool,
    /// Byte offset within the current fragment where JSON closed (inclusive).
    consumed: usize,
}

impl JsonScanner {
    fn new() -> Self {
        Self {
            depth: 0,
            in_string: false,
            escape: false,
            started: false,
            closed: false,
            consumed: 0,
        }
    }

    fn scan(&mut self, fragment: &str) {
        self.consumed = 0;
        if self.closed {
            return;
        }
        let mut byte_pos: usize = 0;
        for ch in fragment.chars() {
            let ch_len = ch.len_utf8();
            if self.closed {
                break;
            }
            if self.in_string {
                if self.escape {
                    self.escape = false;
                } else {
                    match ch {
                        '\\' => self.escape = true,
                        '"' => self.in_string = false,
                        _ => {}
                    }
                }
                byte_pos += ch_len;
                continue;
            }
            match ch {
                '"' => self.in_string = true,
                '{' | '[' => {
                    self.started = true;
                    self.depth += 1;
                }
                '}' | ']' => {
                    self.depth = self.depth.saturating_sub(1);
                    if self.started && self.depth == 0 {
                        self.closed = true;
                        self.consumed = byte_pos + ch_len;
                    }
                }
                _ => {}
            }
            byte_pos += ch_len;
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

// ─── Main Parser ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct IncrementalStreamingParser {
    state: ParserState,
    /// Pending text buffer. Content before `cursor` is logically consumed.
    buf: String,
    /// Read cursor into `buf` — bytes before this are garbage.
    cursor: usize,
    tool_json: String,
    scanner: JsonScanner,
    options: ParserOptions,
    /// Precomputed active markers (built once at construction).
    markers: Vec<&'static str>,
    /// Set of first bytes of all active markers (for fast rejection).
    first_bytes: Vec<u8>,
}

impl IncrementalStreamingParser {
    pub fn new(rule: ParserRule) -> Self {
        Self::with_options(ParserOptions::new(rule))
    }

    pub fn with_options(options: ParserOptions) -> Self {
        let mut markers = Vec::with_capacity(4);
        if options.reasoning_parser {
            if !options.rule.think_start.is_empty() {
                markers.push(options.rule.think_start);
            }
            if !options.rule.think_end.is_empty() {
                markers.push(options.rule.think_end);
            }
        }
        if options.tool_call_parser {
            if !options.rule.tool_start.is_empty() {
                markers.push(options.rule.tool_start);
            }
            if !options.rule.tool_end.is_empty() {
                markers.push(options.rule.tool_end);
            }
        }

        let first_bytes: Vec<u8> = markers.iter().map(|m| m.as_bytes()[0]).collect();

        Self {
            state: ParserState::Normal,
            buf: String::with_capacity(256),
            cursor: 0,
            tool_json: String::with_capacity(512),
            scanner: JsonScanner::new(),
            options,
            markers,
            first_bytes,
        }
    }

    pub fn state(&self) -> ParserState {
        self.state
    }

    pub fn reset(&mut self) {
        self.state = ParserState::Normal;
        self.buf.clear();
        self.cursor = 0;
        self.tool_json.clear();
        self.scanner.reset();
    }

    /// Feed a text delta. Returns zero-copy events borrowing from the internal
    /// buffer. Events are valid until the next `feed()` or `reset()` call.
    pub fn feed(&mut self, delta: &str) -> Vec<ParserEvent<'_>> {
        // Compaction from previous round (deferred to avoid invalidating borrows).
        if self.cursor > COMPACT_THRESHOLD {
            self.buf.drain(..self.cursor);
            self.cursor = 0;
        }

        self.buf.push_str(delta);

        // Use a raw pointer to decouple event lifetime from &mut self.
        // SAFETY: Within a single feed() call, self.buf is never modified
        // (only cursor/state change). Compaction happens at the START of
        // the NEXT feed(), after previous events have been consumed.
        let this = self as *mut Self;
        let mut out: Vec<ParserEvent<'static>> = Vec::with_capacity(8);
        loop {
            let progressed = unsafe {
                match (*this).state {
                    ParserState::Normal => (*this).step_normal(&mut out),
                    ParserState::Reasoning => (*this).step_reasoning(&mut out),
                    ParserState::ToolJson => (*this).step_tool_json(&mut out),
                }
            };
            if !progressed {
                break;
            }
        }
        // SAFETY: events borrow from self.buf which is stable until next feed().
        unsafe { std::mem::transmute::<Vec<ParserEvent<'static>>, Vec<ParserEvent<'_>>>(out) }
    }

    // ─── Buffer Helpers ──────────────────────────────────────────────────

    #[inline]
    fn active(&self) -> &str {
        &self.buf[self.cursor..]
    }

    #[inline]
    fn advance(&mut self, n: usize) {
        self.cursor += n;
    }

    // ─── State Steps ─────────────────────────────────────────────────────

    fn step_normal(&mut self, out: &mut Vec<ParserEvent<'static>>) -> bool {
        if self.cursor >= self.buf.len() {
            return false;
        }

        if let Some((kind, rel_pos, marker_len)) = self.next_marker() {
            if rel_pos > 0 {
                let start = self.cursor;
                let end = start + rel_pos;
                let content = &self.buf[start..end] as *const str;
                out.push(ParserEvent::Content(unsafe { &*content }));
            }
            self.advance(rel_pos + marker_len);
            self.state = match kind {
                MarkerKind::ThinkStart => ParserState::Reasoning,
                MarkerKind::ToolStart => ParserState::ToolJson,
            };
            return true;
        }

        let keep = self.suffix_prefix_len();
        let active_len = self.buf.len() - self.cursor;
        let emit_len = active_len.saturating_sub(keep);
        if emit_len > 0 {
            let start = self.cursor;
            let end = start + emit_len;
            let content = &self.buf[start..end] as *const str;
            out.push(ParserEvent::Content(unsafe { &*content }));
            self.advance(emit_len);
            return true;
        }

        false
    }

    fn step_reasoning(&mut self, out: &mut Vec<ParserEvent<'static>>) -> bool {
        if self.cursor >= self.buf.len() {
            return false;
        }

        let end_tag = self.options.rule.think_end;
        if !end_tag.is_empty() {
            if let Some(rel_pos) = self.active().find(end_tag) {
                if rel_pos > 0 {
                    let start = self.cursor;
                    let end = start + rel_pos;
                    let text = &self.buf[start..end] as *const str;
                    out.push(ParserEvent::Reasoning(unsafe { &*text }));
                }
                self.advance(rel_pos + end_tag.len());
                self.state = ParserState::Normal;
                return true;
            }
        }

        // Fast path: if last byte can't start the end tag, emit everything.
        let active_len = self.buf.len() - self.cursor;
        let last_byte = self.buf.as_bytes()[self.buf.len() - 1];
        if !end_tag.is_empty() && last_byte != end_tag.as_bytes()[0] {
            let start = self.cursor;
            let end = self.buf.len();
            let text = &self.buf[start..end] as *const str;
            out.push(ParserEvent::Reasoning(unsafe { &*text }));
            self.advance(active_len);
            return true;
        }

        let keep = self.suffix_prefix_len();
        let emit_len = active_len.saturating_sub(keep);
        if emit_len > 0 {
            let start = self.cursor;
            let end = start + emit_len;
            let text = &self.buf[start..end] as *const str;
            out.push(ParserEvent::Reasoning(unsafe { &*text }));
            self.advance(emit_len);
            return true;
        }

        false
    }

    fn step_tool_json(&mut self, out: &mut Vec<ParserEvent<'static>>) -> bool {
        match self.options.rule.tool_format {
            ToolCallFormat::Tagged | ToolCallFormat::MiniMaxM2 => self.step_tool_tagged(out),
            ToolCallFormat::RawJson | ToolCallFormat::PrefixedJson => {
                self.step_tool_json_stream(out)
            }
        }
    }

    fn step_tool_tagged(&mut self, out: &mut Vec<ParserEvent<'static>>) -> bool {
        let end_tag = self.options.rule.tool_end;

        if let Some(rel_pos) = self.active().find(end_tag) {
            if rel_pos > 0 {
                let start = self.cursor;
                let end = start + rel_pos;
                self.tool_json.push_str(&self.buf[start..end]);
                let frag = &self.buf[start..end] as *const str;
                out.push(ParserEvent::ToolCallDelta(unsafe { &*frag }));
            }
            self.advance(rel_pos + end_tag.len());

            let payload = std::mem::take(&mut self.tool_json);
            if let Some(tool_call) = parse_tool_call(&payload, self.options.rule.tool_format) {
                out.push(ParserEvent::ToolCall(tool_call));
            }
            self.state = ParserState::Normal;
            return true;
        }

        // Error recovery: buffer too large without closing tag.
        let active_len = self.buf.len() - self.cursor;
        if self.tool_json.len() + active_len > MAX_TOOL_BUF {
            self.recover_tool_as_content(out);
            return true;
        }

        // Fast path: last byte can't start the end tag.
        if active_len > 0 {
            let last_byte = self.buf.as_bytes()[self.buf.len() - 1];
            if !end_tag.is_empty() && last_byte != end_tag.as_bytes()[0] {
                let start = self.cursor;
                let end = self.buf.len();
                self.tool_json.push_str(&self.buf[start..end]);
                let frag = &self.buf[start..end] as *const str;
                out.push(ParserEvent::ToolCallDelta(unsafe { &*frag }));
                self.advance(active_len);
                return true;
            }
        }

        let keep = self.suffix_prefix_len();
        let emit_len = active_len.saturating_sub(keep);
        if emit_len > 0 {
            let start = self.cursor;
            let end = start + emit_len;
            self.tool_json.push_str(&self.buf[start..end]);
            let frag = &self.buf[start..end] as *const str;
            out.push(ParserEvent::ToolCallDelta(unsafe { &*frag }));
            self.advance(emit_len);
            return true;
        }

        false
    }

    fn step_tool_json_stream(&mut self, out: &mut Vec<ParserEvent<'static>>) -> bool {
        if self.cursor >= self.buf.len() {
            return false;
        }

        let frag_start = self.cursor;
        let frag_end = self.buf.len();
        let fragment = &self.buf[frag_start..frag_end];

        self.scanner.scan(fragment);
        self.tool_json.push_str(fragment);

        if self.scanner.closed {
            let consumed = self.scanner.consumed;
            self.cursor = frag_start + consumed;
            self.scanner.reset();

            let payload = std::mem::take(&mut self.tool_json);
            let json_end = payload.len() - (frag_end - frag_start) + consumed;
            let tool_calls = parse_tool_calls_at(&payload, json_end, self.options.rule.tool_format);

            if frag_start < frag_end {
                let frag = &self.buf[frag_start..frag_end] as *const str;
                out.push(ParserEvent::ToolCallDelta(unsafe { &*frag }));
            }
            for tc in tool_calls {
                out.push(ParserEvent::ToolCall(tc));
            }

            self.state = ParserState::Normal;
            return true;
        }

        // Error recovery.
        if self.tool_json.len() > MAX_TOOL_BUF {
            let payload = std::mem::take(&mut self.tool_json);
            self.scanner.reset();
            self.cursor = frag_end;
            // Push payload into buf so we can reference it (rare error path).
            let start = self.buf.len();
            self.buf.push_str(&payload);
            let end = self.buf.len();
            let text = &self.buf[start..end] as *const str;
            out.push(ParserEvent::Content(unsafe { &*text }));
            self.state = ParserState::Normal;
            return true;
        }

        // Normal path: emit the whole fragment as delta.
        self.cursor = frag_end;
        if frag_start < frag_end {
            let frag = &self.buf[frag_start..frag_end] as *const str;
            out.push(ParserEvent::ToolCallDelta(unsafe { &*frag }));
        }
        false
    }

    fn recover_tool_as_content(&mut self, out: &mut Vec<ParserEvent<'static>>) {
        let mut text = std::mem::take(&mut self.tool_json);
        text.push_str(self.active());
        self.cursor = self.buf.len();
        self.scanner.reset();
        // Push into buf so we can return a stable reference (rare error path).
        let start = self.buf.len();
        self.buf.push_str(&text);
        let end = self.buf.len();
        let ptr = &self.buf[start..end] as *const str;
        out.push(ParserEvent::Content(unsafe { &*ptr }));
        self.state = ParserState::Normal;
    }

    // ─── Marker Detection ────────────────────────────────────────────────

    fn next_marker(&self) -> Option<(MarkerKind, usize, usize)> {
        let active = self.active();
        let think = if self.options.reasoning_parser && !self.options.rule.think_start.is_empty() {
            active.find(self.options.rule.think_start)
        } else {
            None
        };
        let tool = if self.options.tool_call_parser && !self.options.rule.tool_start.is_empty() {
            active.find(self.options.rule.tool_start)
        } else {
            None
        };

        match (think, tool) {
            (Some(tp), Some(op)) if tp <= op => Some((
                MarkerKind::ThinkStart,
                tp,
                self.options.rule.think_start.len(),
            )),
            (Some(_), Some(op)) => Some((
                MarkerKind::ToolStart,
                op,
                self.options.rule.tool_start.len(),
            )),
            (Some(tp), None) => Some((
                MarkerKind::ThinkStart,
                tp,
                self.options.rule.think_start.len(),
            )),
            (None, Some(op)) => Some((
                MarkerKind::ToolStart,
                op,
                self.options.rule.tool_start.len(),
            )),
            (None, None) => None,
        }
    }

    /// Longest suffix of active buffer that is a prefix of any active marker.
    fn suffix_prefix_len(&self) -> usize {
        let active = self.active();
        let bytes = active.as_bytes();
        let buf_len = bytes.len();
        if buf_len == 0 {
            return 0;
        }

        let mut keep = 0usize;
        let limit = buf_len.min(self.max_marker_len());
        for start in (buf_len.saturating_sub(limit))..buf_len {
            if !self.first_bytes.contains(&bytes[start]) {
                continue;
            }
            let suffix_len = buf_len - start;
            for marker in &self.markers {
                let mb = marker.as_bytes();
                if suffix_len <= mb.len() && bytes[start..] == mb[..suffix_len] {
                    if suffix_len > keep {
                        keep = suffix_len;
                    }
                    break;
                }
            }
        }
        keep
    }

    #[inline]
    fn max_marker_len(&self) -> usize {
        self.markers.iter().map(|m| m.len()).max().unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkerKind {
    ThinkStart,
    ToolStart,
}

// ─── Tool Call Parsing ───────────────────────────────────────────────────────

fn parse_tool_call(text: &str, format: ToolCallFormat) -> Option<ToolCall> {
    parse_tool_calls(text, format).into_iter().next()
}

fn parse_tool_calls(text: &str, format: ToolCallFormat) -> Vec<ToolCall> {
    match format {
        ToolCallFormat::Tagged => {
            let value: Value = match serde_json::from_str(text) {
                Ok(v) => v,
                Err(_) => return Vec::new(),
            };
            extract_tool_calls(&value)
        }
        ToolCallFormat::PrefixedJson => extract_prefixed_tool_calls(text)
            .map(|(tc, _)| tc)
            .unwrap_or_default(),
        ToolCallFormat::RawJson => {
            let (start, end) = match extract_json_range(text) {
                Some(r) => r,
                None => return Vec::new(),
            };
            let value: Value = match serde_json::from_str(&text[start..end]) {
                Ok(v) => v,
                Err(_) => return Vec::new(),
            };
            extract_tool_calls(&value)
        }
        ToolCallFormat::MiniMaxM2 => parse_minimax_m2(text),
    }
}

/// Parse tool calls using the scanner-provided end position.
fn parse_tool_calls_at(text: &str, json_end: usize, format: ToolCallFormat) -> Vec<ToolCall> {
    let end = json_end.min(text.len());
    match format {
        ToolCallFormat::RawJson => {
            let slice = &text[..end];
            let start = slice
                .char_indices()
                .find(|(_, ch)| !ch.is_whitespace())
                .map(|(i, _)| i)
                .unwrap_or(0);
            match serde_json::from_str(&slice[start..]) {
                Ok(v) => extract_tool_calls(&v),
                Err(_) => Vec::new(),
            }
        }
        ToolCallFormat::PrefixedJson => {
            let slice = &text[..end];
            let json_start = match slice.find('{') {
                Some(p) => p,
                None => return Vec::new(),
            };
            let name = slice[..json_start].trim().to_string();
            if name.is_empty() {
                return Vec::new();
            }
            match serde_json::from_str(&slice[json_start..]) {
                Ok(arguments) => vec![ToolCall { name, arguments }],
                Err(_) => Vec::new(),
            }
        }
        _ => Vec::new(),
    }
}

fn parse_minimax_m2(text: &str) -> Vec<ToolCall> {
    let mut result = Vec::new();
    let mut search = 0;

    while let Some(invoke_start) = text[search..].find("\x3cinvoke") {
        let invoke_start = search + invoke_start;
        let invoke_end = match text[invoke_start..].find("\x3c/invoke\x3e") {
            Some(p) => invoke_start + p + "\x3c/invoke\x3e".len(),
            None => break,
        };
        let block = &text[invoke_start..invoke_end];

        let name = match extract_attr(block, "name") {
            Some(n) => n,
            None => {
                search = invoke_end;
                continue;
            }
        };

        let mut args = serde_json::Map::new();
        let mut offset = 0;
        while let Some(p_start_rel) = block[offset..].find("\x3cparameter") {
            let p_start = offset + p_start_rel;
            let tag_end = match block[p_start..].find('>') {
                Some(p) => p_start + p,
                None => break,
            };
            let tag = &block[p_start..=tag_end];
            let p_name = match extract_attr(tag, "name") {
                Some(n) => n,
                None => {
                    offset = tag_end + 1;
                    continue;
                }
            };
            let close_tag = "\x3c/parameter\x3e";
            let close = match block[tag_end + 1..].find(close_tag) {
                Some(p) => tag_end + 1 + p,
                None => break,
            };
            let param_value = block[tag_end + 1..close].trim();
            let value = serde_json::from_str(param_value)
                .unwrap_or_else(|_| Value::String(param_value.to_string()));
            args.insert(p_name.to_string(), value);
            offset = close + close_tag.len();
        }

        result.push(ToolCall {
            name,
            arguments: Value::Object(args),
        });
        search = invoke_end;
    }

    result
}

fn extract_attr(source: &str, attr: &str) -> Option<String> {
    let idx = source.find(attr)?;
    let after_attr = &source[idx + attr.len()..];
    let after_eq = after_attr.strip_prefix("=\"")?;
    let end = after_eq.find('"')?;
    Some(after_eq[..end].to_string())
}

fn extract_json_range(text: &str) -> Option<(usize, usize)> {
    let start = text
        .char_indices()
        .find(|(_, ch)| !ch.is_whitespace())
        .map(|(index, _)| index)?;
    let mut chars = text[start..].char_indices();
    let (_, first) = chars.next()?;
    let mut depth: u32 = 0;
    let mut in_string = false;
    let mut escape = false;

    match first {
        '{' | '[' => depth = 1,
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
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some((start, start + offset + ch.len_utf8()));
                }
            }
            _ => {}
        }
    }

    None
}

fn extract_prefixed_tool_calls(text: &str) -> Option<(Vec<ToolCall>, usize)> {
    let json_start = text.find('{')?;
    let name = text[..json_start].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let (start, end) = extract_json_range(&text[json_start..])?;
    let arguments: Value =
        serde_json::from_str(&text[json_start + start..json_start + end]).ok()?;
    Some((vec![ToolCall { name, arguments }], json_start + end))
}

fn extract_tool_calls(value: &Value) -> Vec<ToolCall> {
    if let Some(array) = value.as_array() {
        let mut tool_calls = Vec::with_capacity(array.len());
        for item in array {
            tool_calls.extend(extract_tool_calls(item));
        }
        return tool_calls;
    }

    let object = match value.as_object() {
        Some(o) => o,
        None => return Vec::new(),
    };

    if let Some(function) = object.get("function").and_then(Value::as_object) {
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .or_else(|| object.get("name").and_then(Value::as_str));
        if let Some(name) = name {
            let arguments = resolve_arguments(
                function
                    .get("arguments")
                    .or_else(|| object.get("arguments")),
            );
            return vec![ToolCall {
                name: name.to_string(),
                arguments,
            }];
        }
    }

    if let Some(name) = object.get("name").and_then(Value::as_str) {
        let arguments = resolve_arguments(object.get("arguments"));
        return vec![ToolCall {
            name: name.to_string(),
            arguments,
        }];
    }

    Vec::new()
}

fn resolve_arguments(val: Option<&Value>) -> Value {
    match val {
        None => Value::Null,
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
        }
        Some(v) => v.clone(),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Owned version of ParserEvent for test assertions.
    #[derive(Debug, PartialEq)]
    enum Owned {
        Content(String),
        Reasoning(String),
        ToolCall(ToolCall),
        ToolCallDelta(String),
    }

    impl Owned {
        fn from_event(e: &ParserEvent<'_>) -> Self {
            match e {
                ParserEvent::Content(s) => Owned::Content(s.to_string()),
                ParserEvent::Reasoning(s) => Owned::Reasoning(s.to_string()),
                ParserEvent::ToolCall(tc) => Owned::ToolCall(tc.clone()),
                ParserEvent::ToolCallDelta(s) => Owned::ToolCallDelta(s.to_string()),
            }
        }
    }

    fn collect_events(parser: &mut IncrementalStreamingParser, chunks: &[&str]) -> Vec<Owned> {
        let mut events = Vec::new();
        for chunk in chunks {
            for e in parser.feed(chunk) {
                events.push(Owned::from_event(&e));
            }
        }
        events
    }

    #[test]
    fn streams_reasoning_incrementally() {
        let mut parser = IncrementalStreamingParser::new(ParserRule::qwen());
        let events = collect_events(
            &mut parser,
            &[
                "hello ",
                "\x3cthink\x3ean",
                "alyzing",
                "\x3c/think\x3e world",
            ],
        );
        assert_eq!(
            events,
            vec![
                Owned::Content("hello ".into()),
                Owned::Reasoning("an".into()),
                Owned::Reasoning("alyzing".into()),
                Owned::Content(" world".into()),
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
                "\x3ctool_call\x3e{\"name\":\"search\",\"arguments\":",
                "{\"query\":\"Tesla\"}",
                "}\x3c/tool_call\x3e done",
            ],
        );
        assert_eq!(
            events,
            vec![
                Owned::Content("text ".into()),
                Owned::ToolCallDelta("{\"name\":\"search\",\"arguments\":".into()),
                Owned::ToolCallDelta("{\"query\":\"Tesla\"}".into()),
                Owned::ToolCallDelta("}".into()),
                Owned::ToolCall(ToolCall {
                    name: "search".into(),
                    arguments: serde_json::json!({"query":"Tesla"}),
                }),
                Owned::Content(" done".into()),
            ]
        );
    }

    #[test]
    fn keeps_partial_tags_in_buffer() {
        let mut parser = IncrementalStreamingParser::new(ParserRule::qwen());
        let events: Vec<Owned> = parser
            .feed("abc\x3cthi")
            .iter()
            .map(Owned::from_event)
            .collect();
        assert_eq!(events, vec![Owned::Content("abc".into())]);
        assert_eq!(parser.state(), ParserState::Normal);

        let events: Vec<Owned> = parser
            .feed("nk\x3ehello\x3c/think\x3e")
            .iter()
            .map(Owned::from_event)
            .collect();
        assert_eq!(events, vec![Owned::Reasoning("hello".into())]);
    }

    #[test]
    fn parses_nested_function_tool_call_shape() {
        let text = r#"{"function":{"name":"search","arguments":{"query":"mars"}}}"#;
        let tool_call = parse_tool_call(text, ToolCallFormat::Tagged).expect("should parse");
        assert_eq!(tool_call.name, "search");
        assert_eq!(tool_call.arguments, serde_json::json!({"query": "mars"}));
    }

    #[test]
    fn parses_arguments_as_json_string() {
        let text = r#"{"name":"search","arguments":"{\"query\":\"mars\"}"}"#;
        let tool_call = parse_tool_call(text, ToolCallFormat::Tagged).expect("should parse");
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
                "\x3c|python_tag|\x3e{\"name\":\"search\",\"arguments\":",
                "{\"query\":\"Mars\"}} tail",
            ],
        );
        assert_eq!(
            events,
            vec![
                Owned::Content("intro ".into()),
                Owned::ToolCallDelta("{\"name\":\"search\",\"arguments\":".into()),
                Owned::ToolCallDelta("{\"query\":\"Mars\"}} tail".into()),
                Owned::ToolCall(ToolCall {
                    name: "search".into(),
                    arguments: serde_json::json!({"query":"Mars"}),
                }),
                Owned::Content(" tail".into()),
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
                "\x3ctool_calls\x3e{\"name\":\"search\",\"arguments\":",
                "{\"query\":\"saturn\"}}\x3c/tool_calls\x3e after",
            ],
        );
        assert_eq!(
            events,
            vec![
                Owned::Content("before ".into()),
                Owned::ToolCallDelta("{\"name\":\"search\",\"arguments\":".into()),
                Owned::ToolCallDelta("{\"query\":\"saturn\"}}".into()),
                Owned::ToolCall(ToolCall {
                    name: "search".into(),
                    arguments: serde_json::json!({"query":"saturn"}),
                }),
                Owned::Content(" after".into()),
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
                "\x3cminimax:tool_call\x3e\x3cinvoke name=\"search\"\x3e\x3cparameter name=\"query\"\x3e",
                "\"venus\"\x3c/parameter\x3e\x3c/invoke\x3e\x3c/minimax:tool_call\x3e after",
            ],
        );
        assert_eq!(
            events,
            vec![
                Owned::Content("before ".into()),
                Owned::ToolCallDelta(
                    "\x3cinvoke name=\"search\"\x3e\x3cparameter name=\"query\"\x3e".into()
                ),
                Owned::ToolCallDelta("\"venus\"\x3c/parameter\x3e\x3c/invoke\x3e".into()),
                Owned::ToolCall(ToolCall {
                    name: "search".into(),
                    arguments: serde_json::json!({"query":"venus"}),
                }),
                Owned::Content(" after".into()),
            ]
        );
    }

    #[test]
    fn tool_json_stays_buffered_until_close() {
        let mut parser = IncrementalStreamingParser::new(ParserRule::qwen());
        let events: Vec<Owned> = parser
            .feed("\x3ctool_call\x3e{\"name\":\"search\",\"arguments\":")
            .iter()
            .map(Owned::from_event)
            .collect();
        assert_eq!(
            events,
            vec![Owned::ToolCallDelta(
                "{\"name\":\"search\",\"arguments\":".into()
            )]
        );
        assert_eq!(parser.state(), ParserState::ToolJson);
    }

    #[test]
    fn passes_through_reasoning_tags_when_disabled() {
        let mut parser = IncrementalStreamingParser::with_options(ParserOptions {
            rule: ParserRule::qwen(),
            reasoning_parser: false,
            tool_call_parser: true,
        });
        let events = collect_events(
            &mut parser,
            &["hello ", "\x3cthink\x3ekeep", "\x3c/think\x3e world"],
        );
        assert_eq!(
            events,
            vec![
                Owned::Content("hello ".into()),
                Owned::Content("\x3cthink\x3ekeep".into()),
                Owned::Content("\x3c/think\x3e world".into()),
            ]
        );
    }

    #[test]
    fn passes_through_tool_tags_when_disabled() {
        let mut parser = IncrementalStreamingParser::with_options(ParserOptions {
            rule: ParserRule::qwen(),
            reasoning_parser: true,
            tool_call_parser: false,
        });
        let events = collect_events(
            &mut parser,
            &[
                "text ",
                "\x3ctool_call\x3e{\"name\":\"search\",\"arguments\":",
                "{\"query\":\"Tesla\"}}",
                "\x3c/tool_call\x3e done",
            ],
        );
        assert_eq!(
            events,
            vec![
                Owned::Content("text ".into()),
                Owned::Content("\x3ctool_call\x3e{\"name\":\"search\",\"arguments\":".into()),
                Owned::Content("{\"query\":\"Tesla\"}}".into()),
                Owned::Content("\x3c/tool_call\x3e done".into()),
            ]
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut parser = IncrementalStreamingParser::new(ParserRule::qwen());
        let _ = parser.feed("\x3cthink\x3epartial");
        assert_eq!(parser.state(), ParserState::Reasoning);

        parser.reset();
        assert_eq!(parser.state(), ParserState::Normal);

        let events: Vec<Owned> = parser
            .feed("clean text")
            .iter()
            .map(Owned::from_event)
            .collect();
        assert_eq!(events, vec![Owned::Content("clean text".into())]);
    }

    #[test]
    fn error_recovery_on_oversized_tool_buffer() {
        let mut parser = IncrementalStreamingParser::new(ParserRule::qwen());
        let _ = parser.feed("\x3ctool_call\x3e");
        let big = "x".repeat(MAX_TOOL_BUF + 1);
        let events: Vec<Owned> = parser.feed(&big).iter().map(Owned::from_event).collect();

        assert_eq!(parser.state(), ParserState::Normal);
        assert!(events.iter().any(|e| matches!(e, Owned::Content(_))));
    }
}
