#![feature(f16)]

use clap::Parser;
use ellm::config::{Cli, Config};
use ellm::serving;
use ellm::serving::initialize_serving_resources;

fn create_runtime(
    resources: &serving::ServingResources<f16>,
) -> Result<tokio::runtime::Runtime, Box<dyn std::error::Error>> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(resources.api_threads)
        .max_blocking_threads(resources.blocking_threads)
        .enable_all()
        .build()
        .map_err(Into::into)
}

async fn run_server(
    resources: serving::ServingResources<f16>,
) -> Result<(), Box<dyn std::error::Error>> {
    serving::run(
        resources.batch_sequences,
        resources.batch_states,
        resources.shared_state,
        resources.parser_options,
        resources.slot_manager,
    )
    .await?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting backend server...");

    let cli = Cli::parse();
    let config = Config::from_cli(cli)?;
    let resolved_config = config.resolve()?;

    let resources = initialize_serving_resources(&resolved_config)?;

    let rt = create_runtime(&resources)?;

    rt.block_on(async move { run_server(resources).await })?;

    Ok(())
}
