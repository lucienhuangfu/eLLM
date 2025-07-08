use tokenizers::Tokenizer;
use anyhow::{Error as E, Result};


const EOS_TOKEN: &str = "</s>";
fn token() {
    
    let tokenizer_filename = "D:/llama-7b/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

    tokenizer.token_to_id(EOS_TOKEN);
    let prompt = "how are you";
    let tokens = tokenizer.encode(prompt, true).unwrap();
    let ids = tokens.get_ids().to_vec();
    println!("{:?}", ids);
    // let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);
}

#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    use super::*;

    #[test]
    fn test_tokenize() { 
        token();
    }

}