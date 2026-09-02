use ellm::serving;
use ellm::serving::{initialize_serving_resources, ServingConfig};

const MODEL_DIR: &str = "models/Qwen3-Coder-30B-A3B-Instruct";

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
    println!("Starting Qwen3-Coder-30B-A3B-Instruct server...");

    let serving_config = ServingConfig::new(MODEL_DIR.to_string());
    let resources = initialize_serving_resources(&serving_config)?;

    let rt = create_runtime(&resources)?;

    rt.block_on(async move { run_server(resources).await })?;

    Ok(())
}
