use std::f16;
use std::ptr;

// use super::super::super::kernel;
// use super::map_trait::MapTrait;
use crate::compiler::assign::assign;
use crate::init::record::TokenList;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel::generic::sqrt::Sqrt;

// Fuse embedding lookup with RMS normalization
#[derive(Clone)]
pub struct LiftVector<T> {
    token_list_ptr: ConstPtr<TokenList>,
    ptr: MutPtr<T>,
    length: usize,
}

impl<T: Sqrt> LiftVector<T> {
    // Constructor for LookupRMSMap
    pub fn new(token_list_ptr: *const TokenList, ptr: *mut T, length: usize) -> Self {
        Self {
            token_list_ptr: ConstPtr {
                ptr: token_list_ptr,
            },
            ptr: MutPtr { ptr },
            length,
        }
    }

    // Run the map for a given batch size and thread ID
    pub fn run(&self, token_size: usize, decode_size: usize, thread_num: usize, thread_id: usize) {
        let token_list_ptr = self.token_list_ptr.ptr;
        let prefill_decode_size = unsafe { (*token_list_ptr).current_lift_size };
        if let Some((begin, end)) = assign(prefill_decode_size, thread_num, thread_id) {
            unsafe {
                let ptr = self.ptr.ptr;
                let records = &(*token_list_ptr).lift_records;
                for i in begin..end {
                    let lift_index = records[i].lift_index;
                    let prefill_index = records[i].prefill_end_index;
                    let source_ptr = ptr.add(prefill_index * self.length);
                    let destination_ptr = ptr.add(lift_index * self.length);
                    ptr::copy_nonoverlapping(source_ptr, destination_ptr, self.length);
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
