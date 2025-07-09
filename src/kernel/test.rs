use super::kernelf16;
use super::kernelf32;
use std::f16;

pub trait Operator {
    fn add(input_ptr1: *const Self, input_ptr2: *const Self, output_ptr: *mut Self, length: usize);
    fn argmax(input_ptr: *const Self, output_ptr: *mut *const Self, length: usize);
    fn dot_product(
        input_ptr1: *const Self,
        input_ptr2: *const Self,
        length: usize,
        output_ptr: *mut Self,
    );
    fn gelu(input_ptr: *const Self, output_ptr: *mut Self, length: usize);
    fn layer_norm(
        input_ptr: *const Self,
        output_ptr: *mut Self,
        length: usize,
        eps: Self,
        gamma: Self,
        beta: Self,
    );
    fn add_layer_norm(
        input_ptr1: *const Self,
        input_ptr2: *const Self,
        output_ptr: *mut Self,
        length: usize,
        eps: Self,
        gamma: Self,
        beta: Self,
    );
    fn mul(input_ptr1: *const Self, input_ptr2: *const Self, output_ptr: *mut Self, length: usize);
    fn rms_norm(
        input_ptr: *const Self,
        weight: *const Self,
        output_ptr: *mut Self,
        length: usize,
        eps: Self,
    );
    fn add_rms_norm(
        input_ptr1: *const Self,
        input_ptr2: *const Self,
        weight: *const Self,
        output_ptr: *mut Self,
        length: usize,
        eps: Self,
    );
    fn silu(input_ptr: *const Self, output_ptr: *mut Self, length: usize);
    fn scale_softmax(input_ptr: *const Self, output_ptr: *mut Self, length: usize, scale: Self);
}

impl Operator for f32 {
    #[inline]
    fn add(input_ptr1: *const Self, input_ptr2: *const Self, output_ptr: *mut Self, length: usize) {
        kernelf32::add::add(input_ptr1, input_ptr2, output_ptr, length);
    }

    #[inline]
    fn argmax(input_ptr: *const Self, output_ptr: *mut *const Self, length: usize) {
        kernelf32::argmax::argmax(input_ptr, output_ptr, length);
    }

    #[inline]
    fn dot_product(
        input_ptr1: *const Self,
        input_ptr2: *const Self,
        length: usize,
        output_ptr: *mut Self,
    ) {
        kernelf32::dot_product::dot_product(input_ptr1, input_ptr2, length, output_ptr);
    }

    #[inline]
    fn gelu(input_ptr: *const Self, output_ptr: *mut Self, length: usize) {
        kernelf32::gelu::gelu(input_ptr, output_ptr, length);
    }

    #[inline]
    fn layer_norm(
        input_ptr: *const Self,
        output_ptr: *mut Self,
        length: usize,
        eps: Self,
        gamma: Self,
        beta: Self,
    ) {
        kernelf32::layer_norm::layer_norm(input_ptr, output_ptr, length, eps, gamma, beta);
    }

    #[inline]
    fn add_layer_norm(
        input_ptr1: *const Self,
        input_ptr2: *const Self,
        output_ptr: *mut Self,
        length: usize,
        eps: Self,
        gamma: Self,
        beta: Self,
    ) {
        kernelf32::layer_norm::add_layer_norm(
            input_ptr1, input_ptr2, output_ptr, length, eps, gamma, beta,
        );
    }

    #[inline]
    fn mul(input_ptr1: *const Self, input_ptr2: *const Self, output_ptr: *mut Self, length: usize) {
        kernelf32::mul::mul(input_ptr1, input_ptr2, output_ptr, length);
    }

    #[inline]
    fn rms_norm(
        input_ptr: *const Self,
        weight: *const Self,
        output_ptr: *mut Self,
        length: usize,
        eps: Self,
    ) {
        kernelf32::rms_norm::rms_norm(input_ptr, weight, output_ptr, length, eps);
    }

    #[inline]
    fn add_rms_norm(
        input_ptr1: *const Self,
        input_ptr2: *const Self,
        weight: *const Self,
        output_ptr: *mut Self,
        length: usize,
        eps: Self,
    ) {
        kernelf32::rms_norm::add_rms_norm(input_ptr1, input_ptr2, weight, output_ptr, length, eps);
    }

    #[inline]
    fn silu(input_ptr: *const Self, output_ptr: *mut Self, length: usize) {
        kernelf32::silu::silu(input_ptr, output_ptr, length);
    }

    #[inline]
    fn scale_softmax(input_ptr: *const Self, output_ptr: *mut Self, length: usize, scale: Self) {
        kernelf32::softmax::scale_softmax(input_ptr, output_ptr, length, scale);
    }
}

impl Operator for f16 {
    #[inline]
    fn add(input_ptr1: *const Self, input_ptr2: *const Self, output_ptr: *mut Self, length: usize) {
        kernelf16::add::add(input_ptr1, input_ptr2, output_ptr, length);
    }

    #[inline]
    fn argmax(input_ptr: *const Self, output_ptr: *mut *const Self, length: usize) {
        kernelf16::argmax::argmax(input_ptr, output_ptr, length);
    }

    #[inline]
    fn dot_product(
        input_ptr1: *const Self,
        input_ptr2: *const Self,
        length: usize,
        output_ptr: *mut Self,
    ) {
        kernelf16::dot_product::dot_product(input_ptr1, input_ptr2, length, output_ptr);
    }

    #[inline]
    fn gelu(input_ptr: *const Self, output_ptr: *mut Self, length: usize) {
        kernelf16::gelu::gelu(input_ptr, output_ptr, length);
    }

    #[inline]
    fn layer_norm(
        input_ptr: *const Self,
        output_ptr: *mut Self,
        length: usize,
        eps: Self,
        gamma: Self,
        beta: Self,
    ) {
        kernelf16::layer_norm::layer_norm(input_ptr, output_ptr, length, eps, gamma, beta);
    }

    #[inline]
    fn add_layer_norm(
        input_ptr1: *const Self,
        input_ptr2: *const Self,
        output_ptr: *mut Self,
        length: usize,
        eps: Self,
        gamma: Self,
        beta: Self,
    ) {
        kernelf16::layer_norm::add_layer_norm(
            input_ptr1, input_ptr2, output_ptr, length, eps, gamma, beta,
        );
    }

    #[inline]
    fn mul(input_ptr1: *const Self, input_ptr2: *const Self, output_ptr: *mut Self, length: usize) {
        kernelf16::mul::mul(input_ptr1, input_ptr2, output_ptr, length);
    }

    #[inline]
    fn rms_norm(
        input_ptr: *const Self,
        weight: *const Self,
        output_ptr: *mut Self,
        length: usize,
        eps: Self,
    ) {
        kernelf16::rms_norm::rms_norm(input_ptr, weight, output_ptr, length, eps);
    }

    #[inline]
    fn add_rms_norm(
        input_ptr1: *const Self,
        input_ptr2: *const Self,
        weight: *const Self,
        output_ptr: *mut Self,
        length: usize,
        eps: Self,
    ) {
        kernelf16::rms_norm::add_rms_norm(input_ptr1, input_ptr2, weight, output_ptr, length, eps);
    }

    #[inline]
    fn silu(input_ptr: *const Self, output_ptr: *mut Self, length: usize) {
        kernelf16::silu::silu(input_ptr, output_ptr, length);
    }

    #[inline]
    fn scale_softmax(input_ptr: *const Self, output_ptr: *mut Self, length: usize, scale: Self) {
        kernelf16::softmax::scale_softmax(input_ptr, output_ptr, length, scale);
    }
}

pub fn add<T: Operator>(
    input_ptr1: *const T,
    input_ptr2: *const T,
    output_ptr: *mut T,
    length: usize,
) {
    Operator::add(input_ptr1, input_ptr2, output_ptr, length);
}

pub fn argmax<T: Operator>(input_ptr: *const T, output_ptr: *mut *const T, length: usize) {
    Operator::argmax(input_ptr, output_ptr, length);
}

pub fn dot_product<T: Operator>(
    input_ptr1: *const T,
    input_ptr2: *const T,
    length: usize,
    output_ptr: *mut T,
) {
    Operator::dot_product(input_ptr1, input_ptr2, length, output_ptr);
}

pub fn gelu<T: Operator>(input_ptr: *const T, output_ptr: *mut T, length: usize) {
    Operator::gelu(input_ptr, output_ptr, length);
}

pub fn layer_norm<T: Operator>(
    input_ptr: *const T,
    output_ptr: *mut T,
    length: usize,
    eps: T,
    gamma: T,
    beta: T,
) {
    Operator::layer_norm(input_ptr, output_ptr, length, eps, gamma, beta);
}

pub fn add_layer_norm<T: Operator>(
    input_ptr1: *const T,
    input_ptr2: *const T,
    output_ptr: *mut T,
    length: usize,
    eps: T,
    gamma: T,
    beta: T,
) {
    Operator::add_layer_norm(input_ptr1, input_ptr2, output_ptr, length, eps, gamma, beta);
}

pub fn mul<T: Operator>(
    input_ptr1: *const T,
    input_ptr2: *const T,
    output_ptr: *mut T,
    length: usize,
) {
    Operator::mul(input_ptr1, input_ptr2, output_ptr, length);
}

pub fn rms_norm<T: Operator>(
    input_ptr: *const T,
    weight: *const T,
    output_ptr: *mut T,
    length: usize,
    eps: T,
) {
    Operator::rms_norm(input_ptr, weight, output_ptr, length, eps);
}

pub fn add_rms_norm<T: Operator>(
    input_ptr1: *const T,
    input_ptr2: *const T,
    weight: *const T,
    output_ptr: *mut T,
    length: usize,
    eps: T,
) {
    Operator::add_rms_norm(input_ptr1, input_ptr2, weight, output_ptr, length, eps);
}

pub fn silu<T: Operator>(input_ptr: *const T, output_ptr: *mut T, length: usize) {
    Operator::silu(input_ptr, output_ptr, length);
}

pub fn scale_softmax<T: Operator>(
    input_ptr: *const T,
    output_ptr: *mut T,
    length: usize,
    scale: T,
) {
    Operator::scale_softmax(input_ptr, output_ptr, length, scale);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;
    use std::ptr;

    #[test]
    fn test_add() {
        let v1 = [1.0; 19];
        let v2 = [2.0; 19];
        let mut v3 = [0.0; 19];
        let result = [3.0; 19];
        add(v1.as_ptr(), v2.as_ptr(), v3.as_mut_ptr(), v1.len());
        assert_ulps_eq!(v3[..], result[..], max_ulps = 4);
    }

    #[test]
    fn test_argmax() {
        let v: Vec<f32> = vec![2.0, 3.0, 1.0, 5.0, 1.0, 4.0];
        let vptr = v.as_ptr();
        let mut output_ptr = ptr::null();
        argmax(v.as_ptr(), &mut output_ptr, v.len());
        let result = unsafe { vptr.add(3) };
        assert_eq!(result, output_ptr);
    }

    #[test]
    fn test_dot_product() {
        let x1: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let x2: Vec<f32> = (19..37).map(|x| x as f32).collect();
        let mut result = 1.0f32;
        dot_product(x1.as_ptr(), x2.as_ptr(), x1.len(), &mut result);
        assert_eq!(result, 5188.0);
    }

    #[test]
    fn test_gelu() {
        let v1: [f32; 18] = [
            0.3136247992515564,
            -2.6968541145324707,
            0.7449121475219727,
            -0.3058519959449768,
            -1.034066915512085,
            -0.985573410987854,
            -0.5345404744148254,
            -1.3619849681854248,
            0.3012881577014923,
            0.8911539912223816,
            -1.2453598976135254,
            -0.3054046630859375,
            0.5982641577720642,
            -0.21695956587791443,
            -0.0798346996307373,
            -0.7486835718154907,
            0.6165731549263,
            -1.0666285753250122,
        ];
        let mut output = [0.0f32; 18];
        let result: [f32; 18] = [
            0.1954157054424286,
            -0.008965473622083664,
            0.5748836994171143,
            -0.11618322879076004,
            -0.15584588050842285,
            -0.15997932851314545,
            -0.15850451588630676,
            -0.11818332970142365,
            0.18631485104560852,
            0.7249078154563904,
            -0.13285332918167114,
            -0.11606530100107193,
            0.4338094890117645,
            -0.08984798192977905,
            -0.03737737238407135,
            -0.1700376570224762,
            0.4508279263973236,
            -0.15277788043022156,
        ];

        gelu(v1.as_ptr(), output.as_mut_ptr(), v1.len());
        assert_ulps_eq!(output[..], result[..], max_ulps = 4)
    }

    #[test]
    fn test_layer_norm() {
        let a: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let mut b = vec![0.0f32; a.len()];
        layer_norm(a.as_ptr(), b.as_mut_ptr(), a.len(), 1.0e-5f32, 1.0, 0.0);
        let c: Vec<f32> = vec![
            -1.6383557319641113,
            -1.4456080198287964,
            -1.2528603076934814,
            -1.060112476348877,
            -0.8673648238182068,
            -0.6746170520782471,
            -0.48186933994293213,
            -0.2891216278076172,
            -0.09637391567230225,
            0.0963737964630127,
            0.28912150859832764,
            0.48186933994293213,
            0.6746169328689575,
            0.867364764213562,
            1.0601123571395874,
            1.252860188484192,
            1.4456080198287964,
            1.6383556127548218,
        ];
        assert_ulps_eq!(&b[..], &c[..], max_ulps = 4);
    }

    #[test]
    fn test_add_layer_norm() {
        let v1: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let v2: Vec<f32> = (19..37).map(|x| x as f32).collect();
        let mut v3 = vec![0.0f32; v1.len()];
        add_layer_norm(
            v1.as_ptr(),
            v2.as_ptr(),
            v3.as_mut_ptr(),
            v1.len(),
            1.0e-5f32,
            1.0,
            0.0,
        );
        let result: [f32; 18] = [
            -1.63835608959198,
            -1.445608377456665,
            -1.2528605461120605,
            -1.060112714767456,
            -0.8673651218414307,
            -0.6746172904968262,
            -0.4818694591522217,
            -0.2891216278076172,
            -0.0963737964630127,
            0.0963737964630127,
            0.2891216278076172,
            0.4818694591522217,
            0.6746170520782471,
            0.8673651218414307,
            1.060112714767456,
            1.2528603076934814,
            1.445608377456665,
            1.6383559703826904,
        ];
        assert_ulps_eq!(v3[..], result, max_ulps = 4);
    }

    #[test]
    fn test_mul() {
        let v1 = [3.0; 19];
        let v2 = [2.0; 19];
        let mut v3 = [0.0; 19];
        let result = [6.0; 19];
        mul(v1.as_ptr(), v2.as_ptr(), v3.as_mut_ptr(), v1.len());
        assert_ulps_eq!(v3[..], result[..], max_ulps = 4);
    }

    #[test]
    fn test_rms_norm() {
        let v1: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let g = [1.0f32; 18];
        let mut output = [0.0f32; 18];
        rms_norm(v1.as_ptr(), g.as_ptr(), output.as_mut_ptr(), v1.len(), 1e-6);
        let result = [
            0.09238425642251968,
            0.18476851284503937,
            0.27715277671813965,
            0.36953702569007874,
            0.4619212746620178,
            0.5543055534362793,
            0.646689772605896,
            0.7390740513801575,
            0.831458330154419,
            0.9238425493240356,
            1.0162267684936523,
            1.1086111068725586,
            1.2009953260421753,
            1.293379545211792,
            1.3857638835906982,
            1.478148102760315,
            1.5705323219299316,
            1.662916660308838,
        ];

        assert_ulps_eq!(output[..], result, max_ulps = 4);
    }

    #[test]
    fn test_add_rms_norm() {
        let v1: Vec<f32> = (0..18).map(|x| x as f32).collect();
        let v2 = [1.0f32; 18];
        let weight = [1.0f32; 18];
        let mut output = [0.0f32; 18];
        add_rms_norm(
            v1.as_ptr(),
            v2.as_ptr(),
            weight.as_ptr(),
            output.as_mut_ptr(),
            v1.len(),
            1e-6,
        );
        let result = [
            0.09238425642251968,
            0.18476851284503937,
            0.27715277671813965,
            0.36953702569007874,
            0.4619212746620178,
            0.5543055534362793,
            0.646689772605896,
            0.7390740513801575,
            0.831458330154419,
            0.9238425493240356,
            1.0162267684936523,
            1.1086111068725586,
            1.2009953260421753,
            1.293379545211792,
            1.3857638835906982,
            1.478148102760315,
            1.5705323219299316,
            1.662916660308838,
        ];

        assert_ulps_eq!(output[..], result, max_ulps = 4);
    }

    #[test]
    fn test_silu() {
        let v1: [f32; 19] = [
            2.1671206951141357,
            1.4490455389022827,
            -2.002431631088257,
            0.5662149786949158,
            0.3909946382045746,
            0.9437483549118042,
            -0.37030690908432007,
            0.7542704939842224,
            0.5875813961029053,
            1.6026240587234497,
            2.2485475540161133,
            -0.6622593402862549,
            -0.0015666020335629582,
            -0.5069465041160583,
            -0.37254711985588074,
            0.4420417249202728,
            -0.9305257201194763,
            0.5145581364631653,
            0.6260590553283691,
        ];
        let mut output = [0.0f32; 19];
        silu(v1.as_ptr(), output.as_mut_ptr(), v1.len());
        let result = [
            1.9444659948349,
            1.1735117435455322,
            -0.23818494379520416,
            0.36118248105049133,
            0.23323695361614227,
            0.6793630719184875,
            -0.15125809609889984,
            0.5129857659339905,
            0.3777032196521759,
            1.3339999914169312,
            2.033867835998535,
            -0.22532200813293457,
            -0.0007826874498277903,
            -0.1905660629272461,
            -0.15197153389453888,
            0.269090861082077,
            -0.2631694972515106,
            0.32204875349998474,
            0.4079371392726898,
        ];

        assert_ulps_eq!(output[..], result, max_ulps = 4);
    }

    #[test]
    fn test_scale_softmax() {
        let v1: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; v1.len()];
        scale_softmax(v1.as_ptr(), output.as_mut_ptr(), v1.len(), 0.65);
        let result: [f32; 18] = [
            7.5933926382276695e-06,
            1.4545462363457773e-05,
            2.7862415663548745e-05,
            5.337157563189976e-05,
            0.00010223548451904207,
            0.00019583618268370628,
            0.0003751322510652244,
            0.0007185811409726739,
            0.0013764717150479555,
            0.0026366880629211664,
            0.005050681531429291,
            0.009674787521362305,
            0.01853245310485363,
            0.035499654710292816,
            0.06800107657909393,
            0.13025879859924316,
            0.24951595067977905,
            0.4779582619667053,
        ];
        assert_ulps_eq!(output[..], result, max_ulps = 4);
    }
}
