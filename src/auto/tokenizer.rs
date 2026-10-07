use std::collections::HashMap as StdHashMap;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::Result;
use serde::Deserialize;
use tiktoken_rs::CoreBPE;

use super::chat_template::ChatTemplate;

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    added_tokens: Option<Vec<TokenizerToken>>,
    pre_tokenizer: Option<TokenizerPreTokenizer>,
    model: TokenizerModel,
}

#[derive(Debug, Deserialize)]
struct TokenizerToken {
    #[serde(default)]
    id: Option<u32>,
    content: String,
    special: bool,
}

#[derive(Debug, Deserialize)]
struct TokenizerPreTokenizer {
    #[serde(default)]
    pretokenizers: Vec<TokenizerPreTokenizerItem>,
}

#[derive(Debug, Deserialize)]
struct TokenizerPreTokenizerItem {
    #[serde(default)]
    pattern: Option<TokenizerSplitPattern>,
}

#[derive(Debug, Deserialize)]
struct TokenizerSplitPattern {
    #[serde(rename = "Regex")]
    regex: String,
}

#[derive(Debug, Deserialize)]
struct TokenizerModel {
    vocab: StdHashMap<String, u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigJson {
    added_tokens_decoder: Option<StdHashMap<String, TokenizerToken>>,
    additional_special_tokens: Option<Vec<String>>,
    eos_token: Option<TokenField>,
    pad_token: Option<TokenField>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenField {
    String(String),
    Object { content: String },
}

fn build_bytelevel_decoder() -> StdHashMap<char, u8> {
    let mut bs: Vec<u32> = (33u32..=126u32).collect();
    bs.extend(161u32..=172u32);
    bs.extend(174u32..=255u32);

    let mut cs = bs.clone();
    let mut used = [false; 256];
    for &b in &bs {
        used[b as usize] = true;
    }

    let mut next = 0u32;
    for b in 0u32..=255u32 {
        if !used[b as usize] {
            bs.push(b);
            cs.push(256 + next);
            next += 1;
        }
    }

    bs.into_iter()
        .zip(cs)
        .filter_map(|(byte, mapped)| char::from_u32(mapped).map(|ch| (ch, byte as u8)))
        .collect()
}

fn decode_bytelevel_token(token: &str, decoder: &StdHashMap<char, u8>) -> Result<Vec<u8>, String> {
    token
        .chars()
        .map(|ch| {
            decoder
                .get(&ch)
                .copied()
                .ok_or_else(|| format!("Unsupported byte-level token char: {:?}", ch))
        })
        .collect()
}

pub fn load_tiktoken(
    tokenizer_json_path: &str,
    tokenizer_config_json_path: &str,
) -> Result<CoreBPE, String> {
    let content = fs::read_to_string(tokenizer_json_path).map_err(|e| {
        format!(
            "Unable to read tokenizer json {}: {}",
            tokenizer_json_path, e
        )
    })?;
    let parsed: TokenizerJson = serde_json::from_str(&content).map_err(|e| {
        format!(
            "Unable to parse tokenizer json {}: {}",
            tokenizer_json_path, e
        )
    })?;

    let tokenizer_config_content = fs::read_to_string(tokenizer_config_json_path).map_err(|e| {
        format!(
            "Unable to read tokenizer config json {}: {}",
            tokenizer_config_json_path, e
        )
    })?;
    let tokenizer_config: TokenizerConfigJson = serde_json::from_str(&tokenizer_config_content)
        .map_err(|e| {
            format!(
                "Unable to parse tokenizer config json {}: {}",
                tokenizer_config_json_path, e
            )
        })?;

    let pattern = parsed
        .pre_tokenizer
        .as_ref()
        .and_then(|pt| {
            pt.pretokenizers
                .iter()
                .find_map(|item| item.pattern.as_ref())
        })
        .map(|p| p.regex.as_str())
        .ok_or_else(|| {
            format!(
                "Unable to find regex pattern in tokenizer json {}",
                tokenizer_json_path
            )
        })?;

    let bytelevel_decoder = {
        static BYTELEVEL_DECODER: OnceLock<StdHashMap<char, u8>> = OnceLock::new();
        BYTELEVEL_DECODER.get_or_init(build_bytelevel_decoder)
    };

    let encoder = parsed.model.vocab.iter().try_fold(
        StdHashMap::with_capacity(parsed.model.vocab.len()),
        |mut acc, (token, id)| {
            let bytes = decode_bytelevel_token(token.as_str(), bytelevel_decoder)?;
            acc.insert(bytes, *id);
            Ok::<_, String>(acc)
        },
    )?;
    let encoder = encoder.into_iter().collect();

    let added_tokens = parsed.added_tokens.unwrap_or_default();
    let mut special_tokens_encoder: StdHashMap<String, u32> =
        StdHashMap::with_capacity(added_tokens.len());
    for token in added_tokens {
        if token.special {
            let id = token.id.ok_or_else(|| {
                format!(
                    "Missing token id for special token {:?} in tokenizer json {}",
                    token.content, tokenizer_json_path
                )
            })?;
            special_tokens_encoder.insert(token.content, id);
        }
    }

    let added_tokens_decoder = tokenizer_config.added_tokens_decoder.unwrap_or_default();
    let mut special_token_ids_by_content = StdHashMap::with_capacity(added_tokens_decoder.len());
    for (token_id, token) in added_tokens_decoder {
        if !token.special {
            continue;
        }

        let parsed_id = token_id.parse::<u32>().map_err(|e| {
            format!(
                "Unable to parse special token id {} in tokenizer config json {}: {}",
                token_id, tokenizer_config_json_path, e
            )
        })?;

        special_token_ids_by_content.insert(token.content.clone(), parsed_id);
        special_tokens_encoder
            .entry(token.content)
            .or_insert(parsed_id);
    }

    let mut insert_from_vocab_or_config = |token: &str| {
        if let Some(id) = parsed.model.vocab.get(token) {
            special_tokens_encoder
                .entry(token.to_owned())
                .or_insert(*id);
            return;
        }

        if let Some(id) = special_token_ids_by_content.get(token) {
            special_tokens_encoder
                .entry(token.to_owned())
                .or_insert(*id);
        }
    };

    for token in tokenizer_config
        .additional_special_tokens
        .unwrap_or_default()
    {
        insert_from_vocab_or_config(&token);
    }

    if let Some(token) = tokenizer_config.eos_token {
        let token = match token {
            TokenField::String(value) => value,
            TokenField::Object { content } => content,
        };
        insert_from_vocab_or_config(&token);
    }
    if let Some(token) = tokenizer_config.pad_token {
        let token = match token {
            TokenField::String(value) => value,
            TokenField::Object { content } => content,
        };
        insert_from_vocab_or_config(&token);
    }

    let special_tokens_encoder = special_tokens_encoder.into_iter().collect();

    CoreBPE::new(encoder, special_tokens_encoder, pattern).map_err(|e| {
        format!(
            "Unable to initialize tiktoken from tokenizer json {}: {}",
            tokenizer_json_path, e
        )
    })
}

/// `AutoTokenizer`：对齐 HuggingFace `tokenization_auto.AutoTokenizer.from_pretrained`
/// 的惯用门面。Python 版按 `model_type` 经 `TOKENIZER_MAPPING` 动态分发到具体
/// tokenizer 类；eLLM 的分词统一走 tiktoken（`load_tiktoken`），无需注册表，
/// 这里把“给定模型目录、自动拼标准文件名、加载 BPE 与 chat template”收敛为
/// 单一入口，并暴露 `encode` / `decode` / `apply_chat_template` 委托方法，
/// 语义与 HF 的 `tokenizer(...)` / `tokenizer.apply_chat_template(...)` 对齐。
///
/// 仅需底层 BPE 而不含 chat template 时，可继续使用导出的 `load_tiktoken`。
pub struct AutoTokenizer {
    tokenizer: CoreBPE,
    chat_template: ChatTemplate,
    tokenizer_json_path: String,
    tokenizer_config_path: String,
    chat_template_path: String,
}

impl AutoTokenizer {
    /// 从模型目录加载。目录下需存在 `tokenizer.json` 与 `tokenizer_config.json`；
    /// chat template 优先读 `chat_template.jinja`，缺失时回退到
    /// `tokenizer_config.json` 内嵌的 `chat_template`（由 `ChatTemplate` 负责）。
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let dir = model_dir.as_ref();
        let tokenizer_json_path = dir.join("tokenizer.json");
        let tokenizer_config_path = dir.join("tokenizer_config.json");
        let chat_template_path = dir.join("chat_template.jinja");

        let tokenizer_json = path_str(&tokenizer_json_path)?;
        let tokenizer_config = path_str(&tokenizer_config_path)?;
        let chat_template_file = path_str(&chat_template_path)?;

        let tokenizer =
            load_tiktoken(tokenizer_json, tokenizer_config).map_err(anyhow::Error::msg)?;
        let chat_template = ChatTemplate::from_model_files(chat_template_file, tokenizer_config)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;

        Ok(Self {
            tokenizer,
            chat_template,
            tokenizer_json_path: tokenizer_json.to_owned(),
            tokenizer_config_path: tokenizer_config.to_owned(),
            chat_template_path: chat_template_file.to_owned(),
        })
    }

    /// 底层 tiktoken BPE。
    pub fn tokenizer(&self) -> &CoreBPE {
        &self.tokenizer
    }

    /// 底层 chat template。
    pub fn chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }

    /// 编码文本（含特殊 token），等价 HF `tokenizer(text).input_ids`。
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode_with_special_tokens(text)
    }

    /// 解码 token id 序列，等价 HF `tokenizer.decode(ids)`。
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids)
            .map_err(|e| anyhow::Error::msg(e.to_string()))
    }

    /// 渲染对话模板，等价 HF `tokenizer.apply_chat_template(...)`。
    pub fn apply_chat_template(
        &self,
        messages: &[(&str, &str)],
        add_generation_prompt: bool,
    ) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.chat_template
            .apply_chat_template(messages, add_generation_prompt)
    }

    pub fn tokenizer_json_path(&self) -> &str {
        &self.tokenizer_json_path
    }

    pub fn tokenizer_config_path(&self) -> &str {
        &self.tokenizer_config_path
    }

    pub fn chat_template_path(&self) -> &str {
        &self.chat_template_path
    }
}

fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow::anyhow!("path is not valid UTF-8: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN3_TOKENIZER_JSON_PATH: &str =
        "./checkpoints/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json";
    const QWEN3_TOKENIZER_CONFIG_JSON_PATH: &str =
        "./checkpoints/Qwen3-Coder-30B-Instruct/tokenizer_config.json";

    #[test]
    fn test_load_qwen3_tokenizer_json() {
        let tokenizer =
            match load_tiktoken(QWEN3_TOKENIZER_JSON_PATH, QWEN3_TOKENIZER_CONFIG_JSON_PATH) {
                Ok(tokenizer) => tokenizer,
                Err(e) => {
                    eprintln!(
                        "Skip: qwen3 tokenizer json is not loadable in this environment: {}",
                        e
                    );
                    return;
                }
            };

        let text = "<|im_start|>user\nhello<|im_end|>";
        let token_ids = tokenizer.encode_with_special_tokens(text);
        let pieces = tokenizer
            .split_by_token(text, true)
            .expect("split_by_token failed");

        assert!(!token_ids.is_empty());
        assert_eq!(token_ids.len(), pieces.len());

        for (idx, (token_id, piece)) in token_ids.iter().zip(pieces.iter()).enumerate() {
            println!("token[{idx}] id={token_id}, piece={piece:?}");
        }

        let decoded = tokenizer.decode(&token_ids).expect("decode failed");
        println!("decoded={decoded:?}");
        assert!(decoded.contains("hello"));
    }
}
