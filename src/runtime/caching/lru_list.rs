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

    /// 测试 LRU 列表的基本初始化
    /// 验证: 新建的列表为空，长度为 0
    #[test]
    fn test_lru_list_basic() {
        let mut list = LruList::new();
        
        // 验证空列表状态
        assert!(list.is_empty(), "New list should be empty");
        assert_eq!(list.len(), 0, "New list should have length 0");
        
        // 验证弹出空列表返回 None
        assert!(list.pop_back().is_none(), "Pop from empty list should return None");
    }

    /// 测试向列表尾部添加元素
    /// 验证: 添加元素后列表长度增加，索引正确分配
    #[test]
    fn test_push_back() {
        let mut list = LruList::new();
        
        // 添加第一个元素
        let idx1 = list.push_back("dialogue-1".to_string());
        assert_eq!(idx1, 0, "First element should get index 0");
        assert_eq!(list.len(), 1, "List should have 1 element");
        assert!(!list.is_empty(), "List should not be empty");
        
        // 添加第二个元素
        let idx2 = list.push_back("dialogue-2".to_string());
        assert_eq!(idx2, 1, "Second element should get index 1");
        assert_eq!(list.len(), 2, "List should have 2 elements");
        
        // 添加第三个元素
        let idx3 = list.push_back("dialogue-3".to_string());
        assert_eq!(idx3, 2, "Third element should get index 2");
        assert_eq!(list.len(), 3, "List should have 3 elements");
    }

    /// 测试从列表中移除中间元素
    /// 验证: 移除中间元素后，前后元素正确连接
    #[test]
    fn test_remove_middle() {
        let mut list = LruList::new();
        list.push_back("dialogue-1".to_string());  // idx 0
        list.push_back("dialogue-2".to_string());  // idx 1
        list.push_back("dialogue-3".to_string());  // idx 2
        
        // 移除中间元素 (dialogue-2)
        list.remove(1);
        assert_eq!(list.len(), 2, "List should have 2 elements after removal");
        
        // 验证剩余元素顺序正确
        let result1 = list.pop_back();
        assert_eq!(result1, Some("dialogue-3".to_string()), "First pop should return dialogue-3");
        
        let result2 = list.pop_back();
        assert_eq!(result2, Some("dialogue-1".to_string()), "Second pop should return dialogue-1");
        
        assert!(list.is_empty(), "List should be empty after all pops");
    }

    /// 测试移除头部元素
    /// 验证: 移除头部元素后，新头部正确设置
    #[test]
    fn test_remove_head() {
        let mut list = LruList::new();
        list.push_back("dialogue-1".to_string());  // idx 0
        list.push_back("dialogue-2".to_string());  // idx 1
        list.push_back("dialogue-3".to_string());  // idx 2
        
        // 移除头部元素
        list.remove(0);
        assert_eq!(list.len(), 2, "List should have 2 elements");
        
        // 验证顺序正确
        let result1 = list.pop_back();
        assert_eq!(result1, Some("dialogue-3".to_string()));
        
        let result2 = list.pop_back();
        assert_eq!(result2, Some("dialogue-2".to_string()));
    }

    /// 测试移除尾部元素
    /// 验证: 移除尾部元素后，新尾部正确设置
    #[test]
    fn test_remove_tail() {
        let mut list = LruList::new();
        list.push_back("dialogue-1".to_string());  // idx 0
        list.push_back("dialogue-2".to_string());  // idx 1
        list.push_back("dialogue-3".to_string());  // idx 2
        
        // 移除尾部元素
        list.remove(2);
        assert_eq!(list.len(), 2, "List should have 2 elements");
        
        // 验证顺序正确
        let result1 = list.pop_back();
        assert_eq!(result1, Some("dialogue-2".to_string()));
        
        let result2 = list.pop_back();
        assert_eq!(result2, Some("dialogue-1".to_string()));
    }

    /// 测试弹出尾部元素
    /// 验证: pop_back 返回尾部元素并移除它
    #[test]
    fn test_pop_back() {
        let mut list = LruList::new();
        list.push_back("dialogue-1".to_string());
        list.push_back("dialogue-2".to_string());
        
        // 弹出第一个元素
        let result = list.pop_back();
        assert_eq!(result, Some("dialogue-2".to_string()), "Should pop dialogue-2");
        assert_eq!(list.len(), 1, "List should have 1 element left");
        
        // 弹出第二个元素
        let result = list.pop_back();
        assert_eq!(result, Some("dialogue-1".to_string()), "Should pop dialogue-1");
        assert_eq!(list.len(), 0, "List should be empty");
        
        // 弹出空列表
        let result = list.pop_back();
        assert!(result.is_none(), "Pop from empty list should return None");
    }

    /// 测试索引重用机制
    /// 验证: 移除的元素索引被正确重用
    #[test]
    fn test_index_reuse() {
        let mut list = LruList::new();
        
        // 添加两个元素
        let idx1 = list.push_back("dialogue-1".to_string());
        let idx2 = list.push_back("dialogue-2".to_string());
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        
        // 移除第一个元素
        list.remove(idx1);
        assert_eq!(list.len(), 1, "List should have 1 element");
        
        // 添加新元素，应该重用已释放的索引
        let idx3 = list.push_back("dialogue-3".to_string());
        assert_eq!(idx3, idx1, "New element should reuse the freed index");
        assert_eq!(list.len(), 2, "List should have 2 elements");
    }

    /// 测试连续添加和移除的混合场景
    /// 验证: 索引重用和链表连接都正确工作
    #[test]
    fn test_mixed_operations() {
        let mut list = LruList::new();
        
        // 初始添加
        let idx0 = list.push_back("d0".to_string());  // 0
        let idx1 = list.push_back("d1".to_string());  // 1
        let idx2 = list.push_back("d2".to_string());  // 2
        let idx3 = list.push_back("d3".to_string());  // 3
        assert_eq!(list.len(), 4);
        
        // 移除中间元素
        list.remove(idx1);
        list.remove(idx2);
        assert_eq!(list.len(), 2);
        
        // 添加新元素，应该重用空闲索引
        let idx4 = list.push_back("d4".to_string());  // 应该重用 1 或 2
        let idx5 = list.push_back("d5".to_string());  // 应该重用另一个
        assert_eq!(list.len(), 4);
        
        // 验证所有元素都能正确弹出
        let results = vec![
            list.pop_back(),
            list.pop_back(),
            list.pop_back(),
            list.pop_back(),
        ];
        
        // 顺序应该是: d5, d4, d3, d0 (d1 和 d2 已被移除)
        assert!(results.contains(&Some("d0".to_string())));
        assert!(results.contains(&Some("d3".to_string())));
        assert!(results.contains(&Some("d4".to_string())));
        assert!(results.contains(&Some("d5".to_string())));
    }

    /// 测试单个元素的移除
    /// 验证: 只有一个元素时移除它，列表变为空
    #[test]
    fn test_remove_single_element() {
        let mut list = LruList::new();
        let idx = list.push_back("single".to_string());
        
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
        
        // 移除唯一元素
        list.remove(idx);
        
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
        assert!(list.pop_back().is_none());
    }

    /// 测试多次重用相同索引
    /// 验证: 索引可以被多次重用
    #[test]
    fn test_multiple_index_reuse() {
        let mut list = LruList::new();
        
        // 添加并移除，重复多次
        for i in 0..5 {
            let idx = list.push_back(format!("dialogue-{}", i));
            assert_eq!(idx, 0, "Index should be reused each time");
            list.remove(idx);
            assert_eq!(list.len(), 0);
        }
        
        // 最终状态应该是空的
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }
}
