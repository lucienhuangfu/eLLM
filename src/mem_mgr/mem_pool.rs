use once_cell::sync::Lazy;
use regex::{Regex, RegexSet};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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

#[derive(Debug, Default)]
pub struct ScratchPool<T> {
    blocks: HashMap<String, AlignedBox<T>>,
}

unsafe impl<T: Send> Send for MemPool<T> {}
unsafe impl<T: Sync> Sync for MemPool<T> {}

// 全局单例支持
static GLOBAL_MEM_POOL_F32: Lazy<Mutex<Option<MemPool<f32>>>> = Lazy::new(|| Mutex::new(None));
static GLOBAL_MEM_POOL_F16: Lazy<Mutex<Option<MemPool<f16>>>> = Lazy::new(|| Mutex::new(None));
static GLOBAL_SCRATCH_BOOL: Lazy<Mutex<ScratchPool<bool>>> =
    Lazy::new(|| Mutex::new(ScratchPool::new()));
static GLOBAL_SCRATCH_USIZE: Lazy<Mutex<ScratchPool<usize>>> =
    Lazy::new(|| Mutex::new(ScratchPool::new()));

pub trait GlobalMemPool {
    fn init_global(parameters: HashMap<String, Vec<Self>>)
    where
        Self: Sized + Default + Copy;
    fn with_global<F, R>(f: F) -> R
    where
        Self: Sized + Default + Copy,
        F: FnOnce(&mut MemPool<Self>) -> R;
}

pub trait GlobalScratchPool: Copy + Default {
    fn with_scratch_pool<F, R>(f: F) -> R
    where
        F: FnOnce(&mut ScratchPool<Self>) -> R;
}

impl GlobalMemPool for f32 {
    fn init_global(parameters: HashMap<String, Vec<f32>>) {
        let mut pool = GLOBAL_MEM_POOL_F32.lock().unwrap();
        *pool = Some(MemPool::new(parameters));
    }

    fn with_global<F, R>(f: F) -> R
    where
        F: FnOnce(&mut MemPool<f32>) -> R,
    {
        let mut pool = GLOBAL_MEM_POOL_F32.lock().unwrap();
        let pool = pool
            .as_mut()
            .expect("Global MemPool not initialized for f32");
        f(pool)
    }
}

impl GlobalMemPool for f16 {
    fn init_global(parameters: HashMap<String, Vec<f16>>) {
        let mut pool = GLOBAL_MEM_POOL_F16.lock().unwrap();
        *pool = Some(MemPool::new(parameters));
    }

    fn with_global<F, R>(f: F) -> R
    where
        F: FnOnce(&mut MemPool<f16>) -> R,
    {
        let mut pool = GLOBAL_MEM_POOL_F16.lock().unwrap();
        let pool = pool
            .as_mut()
            .expect("Global MemPool not initialized for f16");
        f(pool)
    }
}

impl GlobalScratchPool for bool {
    fn with_scratch_pool<F, R>(f: F) -> R
    where
        F: FnOnce(&mut ScratchPool<bool>) -> R,
    {
        let mut pool = GLOBAL_SCRATCH_BOOL.lock().unwrap();
        f(&mut pool)
    }
}

impl GlobalScratchPool for usize {
    fn with_scratch_pool<F, R>(f: F) -> R
    where
        F: FnOnce(&mut ScratchPool<usize>) -> R,
    {
        let mut pool = GLOBAL_SCRATCH_USIZE.lock().unwrap();
        f(&mut pool)
    }
}

impl<T> ScratchPool<T>
where
    T: Copy + Default,
{
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
        }
    }

    pub fn get_init(&mut self, name: &str, len: usize, value: T) -> *mut T {
        assert!(len > 0, "Scratch allocation length must be greater than 0");

        let needs_alloc = self
            .blocks
            .get(name)
            .map_or(true, |block| block.len() < len);

        if needs_alloc {
            self.blocks
                .insert(name.to_string(), AlignedBox::allocate_init(len, value));
        } else {
            self.blocks.get_mut(name).unwrap().as_mut_slice()[..len].fill(value);
        }

        self.blocks.get_mut(name).unwrap().as_mut_ptr()
    }
}

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

    pub fn get_scratch(&mut self, name: &str, len: usize, value: T) -> *mut T {
        self.get_or_allocate_full(name, len, Some(value)).as_mut_ptr()
    }

    fn block_has_capacity(block: &MemoryBlock<T>, size: usize) -> bool {
        match block {
            MemoryBlock::Full(boxed) => boxed.len() >= size,
            MemoryBlock::Sub { size: block_size, .. } => *block_size >= size,
        }
    }

    fn get_existing_if_valid(&mut self, name: &str, size: usize) -> Option<&mut MemoryBlock<T>> {
        if self
            .blocks
            .get(name)
            .map_or(false, |block| Self::block_has_capacity(block, size))
        {
            self.blocks.get_mut(name)
        } else {
            None
        }
    }

    fn insert_full_from_vec(&mut self, name: &str, data: Vec<T>) -> &mut MemoryBlock<T> {
        let len = data.len();
        let mut boxed = AlignedBox::allocate(len);
        boxed.as_mut_slice().copy_from_slice(&data);
        self.blocks
            .insert(name.to_string(), MemoryBlock::Full(Arc::new(boxed)));
        self.blocks.get_mut(name).unwrap()
    }

    fn get_or_allocate_full(
        &mut self,
        name: &str,
        size: usize,
        init: Option<T>,
    ) -> &mut MemoryBlock<T> {
        assert!(size > 0, "Memory pool allocation size must be greater than 0");

        if self.get_existing_if_valid(name, size).is_none() {
            let boxed = AlignedBox::allocate_init(size, init.unwrap_or_default());
            self.blocks
                .insert(name.to_string(), MemoryBlock::Full(Arc::new(boxed)));
        } else if let Some(value) = init {
            self.blocks
                .get_mut(name)
                .unwrap()
                .as_mut_slice()[..size]
                .fill(value);
        }

        self.blocks.get_mut(name).unwrap()
    }

    fn get_block(&mut self, name: &str, shape: &[usize]) -> &mut MemoryBlock<T> {
        let size: usize = shape.iter().product();

        if self.get_existing_if_valid(name, size).is_some() {
            return self.blocks.get_mut(name).unwrap();
        }

        for m in REGEX_SET.matches(name).iter() {
            match m {
                0 | 3..=6 => {
                    return self.get_or_allocate_full(name, size, None);
                }
                1 => {
                    if self.blocks.contains_key(name) {
                        return self.blocks.get_mut(name).unwrap();
                    }

                    match self.parameters.remove(name) {
                        Some(data) => return self.insert_full_from_vec(name, data),
                        None => panic!("Parameter {} not found in parameters map", name),
                    }
                }
                2 => {
                    if let Some(captures) = LAYER_SHARED_REGEX.captures(name) {
                        let key_name = captures.get(1).unwrap().as_str().to_string();
                        return self.get_or_allocate_full(&key_name, size, None);
                    } else {
                        return self.get_or_allocate_full(name, size, None);
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
