use std::ops::Deref;

struct MyType {
    value: f64,
}

impl Deref for MyType {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl FromUsize for MyType {
    fn from_usize(n: usize) -> Self {
        MyType { value: n as f64 }
    }
}

impl Add for MyType {
    type Output = MyType;

    fn add(self, other: MyType) -> MyType {
        MyType {
            value: self.value + other.value,
        }
    }
}
