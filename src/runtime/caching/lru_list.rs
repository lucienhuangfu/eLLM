#[derive(Clone)]
pub struct LruNode {
    pub dialogue_id: String,
    pub prev: Option<usize>,
    pub next: Option<usize>,
}

pub struct LruList {
    nodes: Vec<LruNode>,
    head: Option<usize>,
    tail: Option<usize>,
    free_indices: Vec<usize>,
}

impl LruList {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            head: None,
            tail: None,
            free_indices: Vec::new(),
        }
    }

    pub fn push_back(&mut self, dialogue_id: String) -> usize {
        let index = if let Some(free_idx) = self.free_indices.pop() {
            self.nodes[free_idx] = LruNode {
                dialogue_id,
                prev: self.tail,
                next: None,
            };
            free_idx
        } else {
            let index = self.nodes.len();
            self.nodes.push(LruNode {
                dialogue_id,
                prev: self.tail,
                next: None,
            });
            index
        };

        if let Some(tail) = self.tail {
            self.nodes[tail].next = Some(index);
        } else {
            self.head = Some(index);
        }
        self.tail = Some(index);
        index
    }

    pub fn remove(&mut self, index: usize) {
        let node = self.nodes[index].clone();

        if let Some(prev) = node.prev {
            self.nodes[prev].next = node.next;
        } else {
            self.head = node.next;
        }

        if let Some(next) = node.next {
            self.nodes[next].prev = node.prev;
        } else {
            self.tail = node.prev;
        }

        self.free_indices.push(index);
    }

    pub fn pop_back(&mut self) -> Option<String> {
        let tail = self.tail?;
        let dialogue_id = self.nodes[tail].dialogue_id.clone();
        self.remove(tail);
        Some(dialogue_id)
    }

    pub fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    pub fn len(&self) -> usize {
        self.nodes.len() - self.free_indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_list_basic() {
        let mut list = LruList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_push_back() {
        let mut list = LruList::new();
        let idx1 = list.push_back("dialogue-1".to_string());
        let idx2 = list.push_back("dialogue-2".to_string());
        
        assert_eq!(list.len(), 2);
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
    }

    #[test]
    fn test_remove() {
        let mut list = LruList::new();
        list.push_back("dialogue-1".to_string());
        list.push_back("dialogue-2".to_string());
        list.push_back("dialogue-3".to_string());
        
        list.remove(1);
        assert_eq!(list.len(), 2);
        
        let result = list.pop_back();
        assert_eq!(result, Some("dialogue-3".to_string()));
        
        let result = list.pop_back();
        assert_eq!(result, Some("dialogue-1".to_string()));
        
        assert!(list.is_empty());
    }

    #[test]
    fn test_pop_back() {
        let mut list = LruList::new();
        list.push_back("dialogue-1".to_string());
        list.push_back("dialogue-2".to_string());
        
        let result = list.pop_back();
        assert_eq!(result, Some("dialogue-2".to_string()));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_index_reuse() {
        let mut list = LruList::new();
        let idx1 = list.push_back("dialogue-1".to_string());
        let idx2 = list.push_back("dialogue-2".to_string());
        
        list.remove(idx1);
        
        let idx3 = list.push_back("dialogue-3".to_string());
        assert_eq!(idx3, idx1);
    }
}
