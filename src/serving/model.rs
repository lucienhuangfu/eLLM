use crate::serving::model_setup::GenerationParameters;
use crate::transformer::config::Config;
use crate::transformer::model::Model;
use crate::transformer::rope::RotaryEmbedding;

pub fn create_position_embeddings(config: &Config) -> Vec<f16> {
    RotaryEmbedding::new(
        config.head_dim,
        config.rotary_dim,
        config.max_position_embeddings,
        config.rope_theta as f32,
        config.rope_scaling.clone(),
    )
    .forward::<f16>()
}

pub fn initialize_model(
    config: &Config,
    gen_params: &GenerationParameters,
    position_vec: Vec<f16>,
    chunk_size: usize,
    batch_size: usize,
    sequence_length: usize,
) -> Model<f16> {
    Model::<f16>::with_sampling(
        config,
        position_vec,
        chunk_size,
        sequence_length,
        batch_size,
        gen_params.top_k,
        gen_params.top_k_simd,
        gen_params.top_p,
        gen_params.min_p,
        gen_params.do_sample,
        gen_params.eos_token_id_list.clone(),
    )
}

pub fn run_model_forward(
    model: &mut Model<f16>,
    sequences_ptr: *mut usize,
    batch_temperature_ptr: *mut f16,
) {
    let _ = model.forward(sequences_ptr, batch_temperature_ptr);
}
