use once_cell::sync::Lazy;
use regex::{Regex, RegexSet};
use std::collections::HashMap;
use std::sync::Arc;

use super::allocator::AlignedBox;

static REGEX_SET: Lazy<RegexSet> = Lazy::new(|| {
    RegexSet::new(&[
        r"model\.layers\.\d+\.self_attn\.(k_position|v_position|key_position|value)",
        r".*\.weight",
        r"model\.layers\.\d+\.(.*)",
        r"model.*\.output",
        r"model.*\.input_layernorm",
        r"model.*\.post_attention_layernorm",
        r"model.*\.(q|k|v|o)_proj\.output",
    ])
    .unwrap()
});

static LAYER_SHARED_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"model\.layers\.\d+\.(.*)").unwrap());

#[derive(Debug, Clone)]
enum MemoryBlock<T> {
    Full(Arc<AlignedBox<T>>),
    Sub {
        parent: Arc<AlignedBox<T>>,
        offset: usize,
        size: usize,
    },
}

impl<T> MemoryBlock<T> {
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        match self {
            MemoryBlock::Full(boxed) => boxed.as_ptr(),
            MemoryBlock::Sub { parent, offset, .. } => parent.as_ptr_offset(*offset),
        }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            MemoryBlock::Full(boxed) => Arc::as_ptr(boxed) as *mut T,
            MemoryBlock::Sub { parent, offset, .. } => {
                (parent.as_ptr() as *mut T).wrapping_add(*offset)
            }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            MemoryBlock::Full(boxed) => boxed.len(),
            MemoryBlock::Sub { size, .. } => *size,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match self {
            MemoryBlock::Full(boxed) => boxed.as_slice(),
            MemoryBlock::Sub {
                parent,
                offset,
                size,
            } => unsafe { std::slice::from_raw_parts(parent.as_ptr().add(*offset), *size) },
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            MemoryBlock::Full(boxed) => unsafe {
                std::slice::from_raw_parts_mut(Arc::as_ptr(boxed) as *mut T, boxed.len())
            },
            MemoryBlock::Sub {
                parent,
                offset,
                size,
            } => unsafe {
                std::slice::from_raw_parts_mut(parent.as_ptr() as *mut T, parent.len())
                    .get_mut(*offset..*offset + *size)
                    .unwrap()
            },
        }
    }
}

#[derive(Debug)]
pub struct MemPool<T> {
    blocks: HashMap<String, MemoryBlock<T>>,
    parameters: HashMap<String, Vec<T>>,
}

unsafe impl<T: Send> Send for MemPool<T> {}
unsafe impl<T: Sync> Sync for MemPool<T> {}

impl<T> MemPool<T>
where
    T: Copy + Default,
{
    pub fn new(parameters: HashMap<String, Vec<T>>) -> Self {
        let mut pool = Self {
            blocks: HashMap::new(),
            parameters,
        };

        let mut expert_groups: HashMap<String, Vec<(usize, String)>> = HashMap::new();
        for key in pool.parameters.keys() {
            if key.contains("experts.") && key.ends_with("proj.weight") {
                if let Some((before_experts, after_experts)) = key.split_once("experts.") {
                    if let Some((idx_str, suffix)) = after_experts.split_once('.') {
                        if let Ok(expert_idx) = idx_str.parse::<usize>() {
                            let base_key = format!("{}experts.{}", before_experts, suffix);
                            expert_groups
                                .entry(base_key)
                                .or_default()
                                .push((expert_idx, key.clone()));
                        }
                    }
                }
            }
        }

        for (base_key, mut experts) in expert_groups {
            experts.sort_by_key(|(idx, _)| *idx);
            let mut all_expert_data = Vec::new();
            let mut expert_sizes = Vec::new();
            for (_, key) in &experts {
                if let Some(data) = pool.parameters.remove(key) {
                    expert_sizes.push(data.len());
                    all_expert_data.extend(data);
                }
            }

            let total_len = all_expert_data.len();
            let mut base_box = AlignedBox::allocate(total_len);
            base_box.as_mut_slice().copy_from_slice(&all_expert_data);
            let parent_arc = Arc::new(base_box);

            pool.blocks
                .insert(base_key.clone(), MemoryBlock::Full(parent_arc.clone()));

            let mut offset = 0usize;
            for ((_, key), &size) in experts.iter().zip(expert_sizes.iter()) {
                pool.blocks.insert(
                    key.clone(),
                    MemoryBlock::Sub {
                        parent: parent_arc.clone(),
                        offset,
                        size,
                    },
                );
                offset += size;
            }
        }

        pool
    }

    pub fn get(&mut self, name: &str, shape: &[usize]) -> *mut T {
        self.get_block(name, shape).as_mut_ptr()
    }

    pub fn get_block(&mut self, name: &str, shape: &[usize]) -> &mut MemoryBlock<T> {
        let size: usize = shape.iter().product();

        let exists_and_valid = self.blocks.get(name).map_or(false, |block| {
            if let MemoryBlock::Full(boxed) = block {
                boxed.len() >= size
            } else {
                true
            }
        });

        if exists_and_valid {
            return self.blocks.get_mut(name).unwrap();
        }

        for m in REGEX_SET.matches(name).iter() {
            match m {
                0 | 3..=6 => {
                    let exists_and_valid = self.blocks.get(name).map_or(false, |block| {
                        if let MemoryBlock::Full(boxed) = block {
                            boxed.len() >= size
                        } else {
                            true
                        }
                    });

                    if exists_and_valid {
                        return self.blocks.get_mut(name).unwrap();
                    }

                    let boxed = AlignedBox::allocate_init(size, T::default());
                    let name_clone = name.to_string();
                    self.blocks
                        .insert(name.to_string(), MemoryBlock::Full(Arc::new(boxed)));
                    return self.blocks.get_mut(&name_clone).unwrap();
                }
                1 => {
                    if self.blocks.contains_key(name) {
                        return self.blocks.get_mut(name).unwrap();
                    }

                    match self.parameters.remove(name) {
                        Some(data) => {
                            let len = data.len();
                            let mut boxed = AlignedBox::allocate(len);
                            boxed.as_mut_slice().copy_from_slice(&data);
                            let name_clone = name.to_string();
                            self.blocks
                                .insert(name.to_string(), MemoryBlock::Full(Arc::new(boxed)));
                            return self.blocks.get_mut(&name_clone).unwrap();
                        }
                        None => panic!("Parameter {} not found in parameters map", name),
                    }
                }
                2 => {
                    if let Some(captures) = LAYER_SHARED_REGEX.captures(name) {
                        let key_name = captures.get(1).unwrap().as_str().to_string();
                        let exists_and_valid = self.blocks.get(&key_name).map_or(false, |block| {
                            if let MemoryBlock::Full(boxed) = block {
                                boxed.len() >= size
                            } else {
                                true
                            }
                        });

                        if exists_and_valid {
                            return self.blocks.get_mut(&key_name).unwrap();
                        }

                        let boxed = AlignedBox::allocate_init(size, T::default());
                        let name_clone = key_name.clone();
                        self.blocks
                            .insert(key_name, MemoryBlock::Full(Arc::new(boxed)));
                        return self.blocks.get_mut(&name_clone).unwrap();
                    } else {
                        let exists_and_valid = self.blocks.get(name).map_or(false, |block| {
                            if let MemoryBlock::Full(boxed) = block {
                                boxed.len() >= size
                            } else {
                                true
                            }
                        });

                        if exists_and_valid {
                            return self.blocks.get_mut(name).unwrap();
                        }

                        let boxed = AlignedBox::allocate_init(size, T::default());
                        let name_clone = name.to_string();
                        self.blocks
                            .insert(name.to_string(), MemoryBlock::Full(Arc::new(boxed)));
                        return self.blocks.get_mut(&name_clone).unwrap();
                    }
                }
                _ => panic!(),
            }
        }

        panic!("No matching pattern found for tensor name: {}", name);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_get() {
        let mut parameters = HashMap::new();
        parameters.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            vec![1.0f32, 2.0f32],
        );
        parameters.insert(
            "model.embed_tokens.weight".to_string(),
            vec![3.0f32, 4.0f32],
        );

        let mut pool: MemPool<f32> = MemPool::new(parameters);

        let name1 = String::from("model.layers.0.self_attn.v_proj.weight");
        let name2 = String::from("model.embed_tokens.weight");
        let p1 = pool.get(&name1, &[2]);
        let p2 = pool.get(&name2, &[2]);
        assert_ne!(p1, p2);

        let k = String::from("model.layers.0.self_attn.k_proj.output");
        let v = String::from("model.layers.0.self_attn.v_proj.output");
        let p1 = pool.get(&k, &[2]);
        let p2 = pool.get(&v, &[2]);
        assert_ne!(p1, p2);

        let layer1 = String::from("model.layers.0.input_layernorm");
        let layer2 = String::from("model.layers.1.input_layernorm");
        let p1 = pool.get(&layer1, &[2]);
        let p2 = pool.get(&layer2, &[2]);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_moe_expert_subblocks() {
        let prefix = "model.layers.0.mlp.";
        let mut parameters = HashMap::new();

        parameters.insert(
            format!("{}experts.0.gate_proj.weight", prefix),
            vec![10.0f32, 20.0f32, 30.0f32],
        );
        parameters.insert(
            format!("{}experts.1.gate_proj.weight", prefix),
            vec![40.0f32, 50.0f32, 60.0f32],
        );

        let mut pool: MemPool<f32> = MemPool::new(parameters);

        let base = pool.get(&format!("{}experts.gate_proj.weight", prefix), &[6]);
        assert!(!base.is_null());

        let sub0 = pool.get(&format!("{}experts.0.gate_proj.weight", prefix), &[3]);
        let sub1 = pool.get(&format!("{}experts.1.gate_proj.weight", prefix), &[3]);

        assert_eq!(sub0, base);
        assert_eq!(sub1, unsafe { base.add(3) });

        unsafe {
            assert_eq!(*sub0.add(0), 10.0f32);
            assert_eq!(*sub0.add(2), 30.0f32);
            assert_eq!(*sub1.add(0), 40.0f32);
            assert_eq!(*sub1.add(2), 60.0f32);
        }
    }
}
