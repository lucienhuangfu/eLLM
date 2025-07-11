use std::collections::HashMap;
use regex::{Regex, RegexSet};
use std::ops::{Add, Sub, Div, Mul, AddAssign, Neg };
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{neg_infinity::NegInfinity, exp::Exp};
use crate::kernel::generic::sigmoid::Sigmoid;

use super::allocator::allocate_init;
#[derive(Debug, Clone)]
pub struct Cache <T>{
    pub storge: HashMap<String, *mut T>,
    regex_set: RegexSet,
    layer_regex: Regex
}

impl <T> Cache <T> 
where T: Copy + Default 
{
    pub fn new()  -> Self {
        let regex_set = RegexSet::new(&[
            r"model.*weight",
            r"model\.layer\.\d+\.self_attn\.(key_position_tensor|value_tensor)",
            r"model\.layer\.\d+\.(.*)",
        ]).unwrap();
        let layer_regex = Regex::new(r"model\.layer\.\d+\.(.*)").unwrap();   

        Self {
            storge: HashMap::new(),
            regex_set: regex_set,
            layer_regex: layer_regex
        }
    }

    pub fn get(&mut self, name: &str, size: usize) -> *mut T {
        // Generate keyname from name
        // 1. For parameter end with weight
        // 2. For k, v
        // 3. For intermediate variables of different layers, they share the same keyname.
        // For example, "model.layers.0.abc.def" and "model.layers.1.abc.def" are two intermediate
        // variables of layers. They share the same keyname "abc.def"
        
        for m in self.regex_set.matches(name).iter() {
            let p = match m {
                0  => {// parameters
                    println!("parameters {} ", name);
                    let data_ptr: *mut T = allocate_init(size, T::default());  
                    self.storge.insert(name.to_owned(), data_ptr);
                    data_ptr
                },
                1  => {// kv 
                    println!("kv {}", name);
                    let data_ptr: *mut T = allocate_init(size, T::default());  
                    self.storge.insert(name.to_owned(), data_ptr);
                    data_ptr
                },
                2 => match self.layer_regex.captures(name) { // layer
                    
                    Some(capture) => {
                        let key_name = capture.get(1).unwrap().as_str();
                        println!("layer {} {}", name, key_name);
                        match self.storge.get(key_name) {
                            Some(ptr) => {
                                ptr.clone()
                            },
                            None => {
                                let data_ptr: *mut T = allocate_init(size, T::default());  
                                self.storge.insert(key_name.to_owned(), data_ptr);
                                data_ptr
                            }
                        }
                    }
                    None => panic!()
                },

                _ => panic!()
            };
            return p;
        }
        panic!()
    }
}

// pub static mut cache: Lazy<Cache> = Lazy::new(|| {Cache::new()});

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_get() {
     

        let mut cache: Cache<f32> = Cache::new();
        unsafe {
           
            let name1 = String::from("model.layer.0.self_attn.v_proj.weight");
            let name2 = String::from("model.embd.weight");
            let p1 = cache.get(&name1, 2);
            let p2 = cache.get(&name2, 2);
            // println!("{:?}", p1);
            assert_ne!(p1, p2);
                 
            let k = String::from("model.layer.0.self_attn.key_position_tensor");
            let v = String::from("model.layer.0.self_attn.mapv_tensor");
            let p1 = cache.get(&k, 2);
            let p2 = cache.get(&v, 2);
            // println!("{:?}", p1);
            assert_ne!(p1, p2);
        
        
     
            let layer1 = String::from("model.layer.0.self_attn.cc");
            let layer2 = String::from("model.layer.1.self_attn.cc");
            let p1 = cache.get(&layer1, 2);
            let p2 = cache.get(&layer2, 2);
            // println!("{:?}", p1);
            assert_eq!(p1, p2);

        }
    }
}