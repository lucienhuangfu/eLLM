#![feature(f16)]

use clap::Parser;
use ellm::config::{Cli, Config};
use ellm::runtime::RuntimeContext;
use ellm::serving;
use ellm::serving::initialize_serving_resources;
use ellm::serving::parser::ParserOptions;

fn create_runtime(
    ctx: &RuntimeContext<f16>,
) -> Result<tokio::runtime::Runtime, Box<dyn std::error::Error>> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(ctx.thread_config.api_threads)
        .max_blocking_threads(ctx.thread_config.blocking_threads)
        .enable_all()
        .build()
        .map_err(Into::into)
}

async fn run_server(
    ctx: RuntimeContext<f16>,
    parser_options: ParserOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    serving::run(
        ctx.scheduler,
        parser_options,
        ctx.slot_manager,
    )
    .await?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting backend server...");

    let cli = Cli::parse();
    let config = Config::from_cli(cli)?;
    let resolved_config = config.resolve()?;

    let (ctx, parser_options) = initialize_serving_resources(&resolved_config)?;

    let rt = create_runtime(&ctx)?;

    rt.block_on(async move { run_server(ctx, parser_options).await })?;

    Ok(())
}
