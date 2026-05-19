use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::operators::operator::Operator;

static GLOBAL_OPERATOR_QUEUE_F32: Lazy<Mutex<Vec<Operator<f32>>>> =
    Lazy::new(|| Mutex::new(Vec::new()));
static GLOBAL_OPERATOR_QUEUE_F16: Lazy<Mutex<Vec<Operator<f16>>>> =
    Lazy::new(|| Mutex::new(Vec::new()));

pub trait GlobalOperatorQueue: Copy + PartialOrd {
    fn init_operator_queue();
    fn take_operator_queue() -> Vec<Operator<Self>>;
    fn with_operator_queue<F, R>(f: F) -> R
    where
        F: FnOnce(&mut Vec<Operator<Self>>) -> R;
}

impl GlobalOperatorQueue for f32 {
    fn init_operator_queue() {
        GLOBAL_OPERATOR_QUEUE_F32.lock().unwrap().clear();
    }

    fn take_operator_queue() -> Vec<Operator<f32>> {
        std::mem::take(&mut *GLOBAL_OPERATOR_QUEUE_F32.lock().unwrap())
    }

    fn with_operator_queue<F, R>(f: F) -> R
    where
        F: FnOnce(&mut Vec<Operator<f32>>) -> R,
    {
        let mut queue = GLOBAL_OPERATOR_QUEUE_F32.lock().unwrap();
        f(&mut queue)
    }
}

impl GlobalOperatorQueue for f16 {
    fn init_operator_queue() {
        GLOBAL_OPERATOR_QUEUE_F16.lock().unwrap().clear();
    }

    fn take_operator_queue() -> Vec<Operator<f16>> {
        std::mem::take(&mut *GLOBAL_OPERATOR_QUEUE_F16.lock().unwrap())
    }

    fn with_operator_queue<F, R>(f: F) -> R
    where
        F: FnOnce(&mut Vec<Operator<f16>>) -> R,
    {
        let mut queue = GLOBAL_OPERATOR_QUEUE_F16.lock().unwrap();
        f(&mut queue)
    }
}
