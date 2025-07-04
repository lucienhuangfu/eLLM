use super::super::compiler::reduce::argmax_reduce::ArgmaxReduce;

#[derive(Clone)]
pub struct Llama<T> {

   
}

impl<T> Model<T> 
where T: Copy 
    + Default 
    + Sub<Output = T>
    + Neg<Output = T>
    + Exp
    + NegInfinity
    + Sigmoid<T>
    + Sqrt
    + FromF32
{

    pub fn build(config: &Config, 
        sequences: *mut usize,
        cache: Rc<RefCell<Cache<f16>>>,
        operator_queue: Rc<RefCell<Vec<Operator<f16>>>>,
        ) -> Tensor<f16> {
        let cpu_num = num_cpus::get();
        let word_embedding = Tensor::zeros(vec![config.vocab_size, config.hidden_size], String::from("model.embed_tokens.weight"), cache.clone(), operator_queue.clone());
        let dim =  config.attention_head_size / 2;
        let rope_vec = precompute_freqs_cis(dim, config.max_position_embeddings, 10000.0f32);
        let position_embedding = Tensor::zeros(vec![config.max_position_embeddings, 1, 1, config.attention_head_size], String::from("model.position_embedding.weight"), cache.clone(), operator_queue.clone());
        for i in 0..rope_vec.len() {
            unsafe {
                position_embedding.data.add(i).write(rope_vec[i] as f16);
            }
        }
        let norm_weight = Tensor::zeros(vec![1, config.hidden_size], String::from("model.norm.weight"), cache.clone(), operator_queue.clone());

        let model = Model::<f16>::new(
            config.clone(),
            word_embedding,
            position_embedding,
            norm_weight,
            cpu_num,
            cache.clone(),
            operator_queue.clone(),
        );
        
        let output_tensor = unsafe {
            model.forward(sequences.add(config.batch_size)) 
        };
        output_tensor
    }

}