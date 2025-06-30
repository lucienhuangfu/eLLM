use std::alloc::{self, Layout};
use std::{ptr, mem};

pub fn allocate<T>(length: usize) -> *mut T {
    //原始版本的 "allocate" "allocate_usize"  "allocate_f16" "allocate_ptr" functions 都被整合到了这个allocate function中 Jason 2024.5.15
    unsafe {
        let layout = Layout::from_size_align_unchecked(length * mem::size_of::<T>(), 64);
        // 分配内存
        let mut data_ptr: *mut T = alloc::alloc(layout) as *mut T;
        if data_ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        data_ptr
    }
}
pub fn allocate_init<T>(length: usize, value: T) -> *mut T 
where T: Copy {
    
    let data_ptr: *mut T = allocate(length);
    for i in 0..length {
        unsafe {
            let p = data_ptr.add(i);
            ptr::write(p, value);
        }
    }
    data_ptr
}

/* 
pub fn allocate_init_usize(length: usize, value: usize) -> *mut usize {
    let data_ptr: *mut usize = allocate(length);
    for i in 0..length {
        unsafe {
            let p = data_ptr.add(i);
            ptr::write(p, value);
        }
    }
    data_ptr
}
*/

#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    use super::*;

    #[test]
    fn test1() { 
        let length = 50;
        let value: f32 = 1.2;
        let data_ptr: *mut f32 = allocate_init::<f32>(length, value);
        for i in 0..length {
            unsafe {
                assert_ulps_eq!(*data_ptr.add(i), value, max_ulps=4);
            }
        }
    }
        /*
    #[test]
    fn test2() { 
        let length = 50;
        let value: usize = 1;
        let data_ptr: *mut usize = allocate_init_usize(length, value);
        for i in 0..length {
            unsafe {
                // assert_ulps_eq!(*data_ptr.add(i), value, max_ulps=4);
                assert_eq!(*data_ptr.add(i), value);
            }
        }
    }


    #[test]
    fn test2() {
        let length = 50;
        let num: f32 =1.2;
        let data_ptr: *mut f32 = allocate_init::<f32>(length, num);
        let offset = data_ptr.align_offset(mem::align_of::<f32>()*16);
        if offset == 0 {
        } else if offset < length - 1 {
            println!("offset {}", offset);
            unsafe {
                let _ptr = data_ptr.add(offset).cast::<u16>();
            }
            // *u16_ptr = 0;
        } else {
            // 虽然指针可以通过 `offset` 对齐，但它会指向分配之外
        }
    }
    #[test]
    fn test1() {
        let length = 50;
        let num: f16 =  1.2);
        let data_ptr: *mut f16 = allocate_init::<f16>(length,num);
        let offset = data_ptr.align_offset(mem::align_of::<f16>()*16);
        if offset == 0 {
        } else if offset < length - 1 {
            println!("offset {}", offset);
            unsafe {
                let _ptr = data_ptr.add(offset).cast::<u16>();
            }
            // *u16_ptr = 0;
        } else {
            // 虽然指针可以通过 `offset` 对齐，但它会指向分配之外
        }
    }
    #[test]
    fn test3() {
        let length = 50;
        let data_ptr: *mut f16 = allocate(length);
        let offset = data_ptr.align_offset(mem::align_of::<f16>()*16);

        if offset == 0 {

        } else if offset < length - 1 {
            println!("offset {}", offset);
            unsafe {
                let _ptr = data_ptr.add(offset).cast::<u16>();
            }
            // *u16_ptr = 0;
        } else {
            // 虽然指针可以通过 `offset` 对齐，但它会指向分配之外
        }
    }

    #[test]
    fn test4() {
        let length = 50;
        let data_ptr: *mut u16 = allocate(length);
        let offset = data_ptr.align_offset(mem::align_of::<u16>()*16);

        if offset == 0 {
            println!("offset 0")
        } else if offset < length - 1 {
            println!("offset {}", offset);
            unsafe {
                let _ptr = data_ptr.add(offset).cast::<u16>();
            }
            // *u16_ptr = 0;
        } else {
            // 虽然指针可以通过 `offset` 对齐，但它会指向分配之外
        }
    }

    #[test]
    fn test5() {
        let length = 50;
        let num = 100;
        let data_ptr = allocate_usize_init(length,num);
        let offset = data_ptr.align_offset(mem::align_of::<usize>()*16);
        if offset == 0 {
        } else if offset < length - 1 {
            println!("offset {}", offset);
            unsafe {
                let _ptr = data_ptr.add(offset).cast::<u16>();
            }
            // *u16_ptr = 0;
        } else {
            // 虽然指针可以通过 `offset` 对齐，但它会指向分配之外
        }
    } */
}


