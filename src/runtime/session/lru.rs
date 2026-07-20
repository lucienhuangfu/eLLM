pub(crate) struct LruList {
    slots: Vec<usize>,
}

impl LruList {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            slots: (0..n).collect(),
        }
    }

    pub(crate) fn remove(&mut self, idx: usize) {
        if let Some(pos) = self.slots.iter().position(|&x| x == idx) {
            self.slots.remove(pos);
        }
    }

    pub(crate) fn push_front(&mut self, idx: usize) {
        self.remove(idx);
        self.slots.insert(0, idx);
    }

    pub(crate) fn pop_back(&mut self) -> usize {
        self.slots.pop().expect("LRU list is empty")
    }
}
