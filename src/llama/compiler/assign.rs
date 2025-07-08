// this is a helper function that help locate the partial tasks for the current thread
// parse the range [0, length) evenly into num parts, and return the begin and end of the id-th part
// num is positive, id is 0-indexed, in range [0, num)
// let return tuple be (l, r); elements in range [l, r) belong to the id-th part

// when the length is not a multiple of num, the remain is distributed to the first remain parts
// length
// when length = 10, num = 3, then the parts have 4, 3, 3 elements respectively, the tuples are (0, 4), (4, 7), (7, 10)
// when length = 11, num = 3, then the parts have 4, 4, 3 elements respectively, the tuples are (0, 4), (4, 8), (8, 11)
// when length = 12, num = 3, then the parts have 4, 4, 4 elements respectively, the tuples are (0, 4), (4, 8), (8, 12)
pub fn assign(length: usize, num: usize, id: usize) -> Option<(usize, usize)> {
    debug_assert!(num != 0);
    debug_assert!(id < num);

    if length < (id + 1) {
        return None;
    }

    let (quotient, remainder) = (length / num, length % num);
    // when the length is a multiple of num
    if remainder == 0 {
        let begin = quotient * id;
        let end = begin + quotient;
        return Some((begin, end));
    }

    // when the length is not a multiple of num
    // the remainder is evenlydistributed to the first remainder parts
    if id < remainder {
        let begin = (quotient + 1) * id;
        let end = begin + (quotient + 1);
        return Some((begin, end));
    }

    let begin = quotient * id + remainder;
    let end = begin + quotient;
    return Some((begin, end));
}

#[cfg(test)]
mod test {
    // use std::result;
    // use approx::assert_ulps_eq;
    use super::*;


    // test assign method
    #[test]
    fn test_assign() {

        let length = 5;
        let num = 8;

        let result = assign(length, num, 5);
        assert_eq!(result, None);
        let result = assign(length, num, 4);
        assert_eq!(result, Some((4, 5)));


        let length = 10;
        let num = 3;

        let result = assign(length, num, 0);
        assert_eq!(result, Some((0, 4)));

        let result = assign(length, num, 0);
        assert_eq!(result, Some((0, 4)));
   
        let result = assign(length, num, 1);
        assert_eq!(result, Some((4, 7)));
        
        let result = assign(length, num, 2);
        assert_eq!(result, Some((7, 10)));
        
        let length = 11;
        let num = 3;
        let result = assign(length, num, 0);
        assert_eq!(result, Some((0, 4)));
        
        let result = assign(length, num, 1);
        assert_eq!(result, Some((4, 8)));
        
        let result = assign(length, num, 2);
        assert_eq!(result, Some((8, 11)));
        
        let length = 12;
        let num = 3;
        let result = assign(length, num, 0);
        assert_eq!(result, Some((0, 4)));

        let result = assign(length, num, 1);
        assert_eq!(result, Some((4, 8)));

        let result = assign(length, num, 2);
        assert_eq!(result, Some((8, 12)));
         
    }
}