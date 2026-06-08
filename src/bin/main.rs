#![feature(f16)]

use clap::Parser;

use ellm::serving;
use ellm::serving::{initialize_serving_resources, ServingConfig};

#[derive(Debug, Parser)]
#[command(name = "eLLM serving")]
struct Cli {
    model_dir: String,

    #[arg(
        long = "reasoning-parser",
        default_value_t = true,
        help = "Enable or disable reasoning tag parsing"
    )]
    reasoning_parser: bool,

    #[arg(
        long = "tool-call-parser",
        default_value_t = true,
        help = "Enable or disable tool call tag parsing"
    )]
    tool_call_parser: bool,
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
    println!("Starting backend server...");

    let cli = Cli::parse();
    let mut serving_config = ServingConfig::new(cli.model_dir);
    serving_config.reasoning_parser_enabled = cli.reasoning_parser;
    serving_config.tool_call_parser_enabled = cli.tool_call_parser;
    let resources = initialize_serving_resources(&serving_config)?;

    let rt = create_runtime(&resources)?;

    rt.block_on(async move { run_server(resources).await })?;

    Ok(())
}
