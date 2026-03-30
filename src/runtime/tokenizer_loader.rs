use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::sync::OnceLock;
use tiktoken_rs::CoreBPE;

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    added_tokens: Option<Vec<AddedToken>>,
    pre_tokenizer: Option<PreTokenizer>,
    model: TokenizerModel,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    special: bool,
}

#[derive(Debug, Deserialize)]
struct PreTokenizer {
    #[serde(default)]
    pretokenizers: Vec<SubPreTokenizer>,
}

#[derive(Debug, Deserialize)]
struct SubPreTokenizer {
    #[serde(default)]
    pattern: Option<SplitPattern>,
}

#[derive(Debug, Deserialize)]
struct SplitPattern {
    #[serde(rename = "Regex")]
    regex: String,
}

#[derive(Debug, Deserialize)]
struct TokenizerModel {
    vocab: HashMap<String, u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigJson {
    added_tokens_decoder: Option<HashMap<String, AddedTokenConfig>>,
    additional_special_tokens: Option<Vec<String>>,
    eos_token: Option<TokenField>,
    pad_token: Option<TokenField>,
}

#[derive(Debug, Deserialize)]
struct AddedTokenConfig {
    content: String,
    special: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenField {
    String(String),
    Object { content: String },
}

impl TokenField {
    fn content(self) -> String {
        match self {
            Self::String(value) => value,
            Self::Object { content } => content,
        }
    }
}

fn build_bytelevel_decoder() -> HashMap<char, u8> {
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

fn bytelevel_decoder() -> &'static HashMap<char, u8> {
    static BYTELEVEL_DECODER: OnceLock<HashMap<char, u8>> = OnceLock::new();
    BYTELEVEL_DECODER.get_or_init(build_bytelevel_decoder)
}

fn decode_bytelevel_token(token: &str, decoder: &HashMap<char, u8>) -> Result<Vec<u8>, String> {
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

    let bytelevel_decoder = bytelevel_decoder();

    let encoder = parsed.model.vocab.iter().try_fold(
        HashMap::with_capacity(parsed.model.vocab.len()),
        |mut acc, (token, id)| {
            let bytes = decode_bytelevel_token(token.as_str(), bytelevel_decoder)?;
            acc.insert(bytes, *id);
            Ok::<_, String>(acc)
        },
    )?;
    let encoder = encoder.into_iter().collect();

    let added_tokens = parsed.added_tokens.unwrap_or_default();
    let mut special_tokens_encoder: HashMap<String, u32> =
        HashMap::with_capacity(added_tokens.len());
    for token in added_tokens {
        if token.special {
            special_tokens_encoder.insert(token.content, token.id);
        }
    }

    let added_tokens_decoder = tokenizer_config.added_tokens_decoder.unwrap_or_default();
    let mut special_token_ids_by_content = HashMap::with_capacity(added_tokens_decoder.len());
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
        let token = token.content();
        insert_from_vocab_or_config(&token);
    }
    if let Some(token) = tokenizer_config.pad_token {
        let token = token.content();
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

#[cfg(test)]
mod tests {
    use super::load_tiktoken;

    const QWEN3_TOKENIZER_JSON_PATH: &str = "./models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json";
    const QWEN3_TOKENIZER_CONFIG_JSON_PATH: &str =
        "./models/Qwen3-Coder-30B-A3B-Instruct/tokenizer_config.json";

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

        let decoded = tokenizer.decode(token_ids).expect("decode failed");
        println!("decoded={decoded:?}");
        assert!(decoded.contains("hello"));
    }
}
