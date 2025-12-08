use std::f16;
use std::ptr;

use super::super::super::kernel;
use super::map_trait::MapTrait;
use crate::compiler::assign::assign;
use crate::init::record::LastPrefillList;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel::generic::sqrt::Sqrt;

// Fuse embedding lookup with RMS normalization
#[derive(Clone)]
pub struct LiftVector<T> {
    last_prefill_ptr: ConstPtr<LastPrefillList>,
    ptr: MutPtr<T>,
    length: usize,
}

impl<T: Sqrt> LiftVector<T> {
    // Constructor for LookupRMSMap
    pub fn new(last_prefill_ptr: *const LastPrefillList, ptr: *mut T, length: usize) -> Self {
        Self {
            last_prefill_ptr: ConstPtr {
                ptr: last_prefill_ptr,
            },
            ptr: MutPtr { ptr },
            length,
        }
    }

    // Run the map for a given batch size and thread ID
    pub fn run(&self, token_size: usize, decode_size: usize, thread_num: usize, thread_id: usize) {
        let last_prefill_ptr = self.last_prefill_ptr;
        let prefill_decode_size = last_prefill_ptr.current_size;
        if let Some((begin, end)) = assign(prefill_decode_size, thread_num, thread_id) {
            unsafe {
                let ptr = self.ptr.ptr;
                for i in begin..end {
                    let decode_index = last_prefill_ptr.records[i].decode_index;
                    let prefill_index = last_prefill_ptr.records[i].prefill_index;
                    let source_ptr = ptr.add(prefill_index * self.length);
                    let destination_ptr = ptr.add(decode_index * self.length);
                    ptr::copy_nonoverlapping(destination_ptr, source_ptr, self.length);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_ulps_eq;
}
