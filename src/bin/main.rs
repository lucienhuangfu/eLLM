#![feature(f16)]

use ellm::serving;
use ellm::serving::{initialize_server, ServerConfig};

fn create_runtime(resources: &serving::ServerResources) -> Result<tokio::runtime::Runtime, Box<dyn std::error::Error>> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(resources.worker_threads)
        .max_blocking_threads(resources.async_threads)
        .enable_all()
        .build()
        .map_err(Into::into)
}

async fn run_server(resources: serving::ServerResources) -> Result<(), Box<dyn std::error::Error>> {
    tokio::spawn(async move {
        resources.runner.start().await;
    });

    serving::run(
        resources.batch_sequences,
        resources.batch_states,
        resources.token_counter,
    )
    .await?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting backend server...");

    let server_config = ServerConfig::default();
    let resources = initialize_server(&server_config)?;
    
    let rt = create_runtime(&resources)?;

    rt.block_on(async move {
        run_server(resources).await
    })?;

    Ok(())
}