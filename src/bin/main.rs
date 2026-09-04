use clap::Parser;
use ellm::serving;
use ellm::serving::{initialize_serving_resources, ServingConfig};

const MODEL_DIR: &str = "models/Qwen3-Coder-30B-A3B-Instruct";

#[derive(Debug, Parser)]
#[command(name = "main", about = "Run the eLLM OpenAI-compatible chat server")]
struct Args {
    /// Directory containing the model configuration, tokenizer, and weights.
    #[arg(long, value_name = "DIR", default_value = MODEL_DIR)]
    model_path: String,

    /// Maximum tokens processed in one prefill chunk.
    #[arg(long, value_name = "TOKENS", value_parser = parse_positive_usize)]
    chunk_size: Option<usize>,

    /// Token capacity of each request slot, including prompt and output.
    #[arg(long, value_name = "TOKENS", value_parser = parse_positive_usize)]
    sequence_length: Option<usize>,

    /// Maximum number of concurrent request slots.
    #[arg(long, value_name = "REQUESTS", value_parser = parse_positive_usize)]
    batch_size: Option<usize>,
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{value:?} is not a positive integer"))
        .and_then(|parsed| {
            if parsed == 0 {
                Err("value must be greater than 0".to_string())
            } else {
                Ok(parsed)
            }
        })
}

fn create_runtime(
    resources: &serving::ServingResources,
) -> Result<tokio::runtime::Runtime, Box<dyn std::error::Error>> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(resources.worker_threads)
        .max_blocking_threads(resources.async_threads)
        .enable_all()
        .build()
        .map_err(Into::into)
}

async fn run_server(
    resources: serving::ServingResources,
) -> Result<(), Box<dyn std::error::Error>> {
    tokio::spawn(async move {
        resources.runner.start().await;
    });

    serving::run(
        resources.batch_sequences,
        resources.batch_states,
        resources.token_counter,
        resources.parser_options,
    )
    .await?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("Starting Qwen3-Coder-30B-A3B-Instruct server...");

    let mut serving_config = ServingConfig::new(args.model_path);
    if let Some(chunk_size) = args.chunk_size {
        serving_config.chunk_size = chunk_size;
    }
    if let Some(sequence_length) = args.sequence_length {
        serving_config.sequence_length = sequence_length;
    }
    if let Some(batch_size) = args.batch_size {
        serving_config.batch_size = batch_size;
    }

    println!(
        "Serving config: chunk_size={}, sequence_length={}, batch_size={}",
        serving_config.chunk_size, serving_config.sequence_length, serving_config.batch_size
    );
    let resources = initialize_serving_resources(&serving_config)?;

    let rt = create_runtime(&resources)?;

    rt.block_on(async move { run_server(resources).await })?;

    Ok(())
}
