use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, RwLock};

use crate::runtime::slot_manager::SlotManager;

#[derive(Debug, Clone)]
pub struct DialogueEntry {
    pub dialogue_id: String,
    pub slot_index: usize,
    pub token_count: usize,
    pub last_accessed_at: Instant,
    pub in_lru: bool,
    pub lru_index: Option<usize>,
}

#[derive(Clone)]
struct LruNode {
    dialogue_id: String,
    prev: Option<usize>,
    next: Option<usize>,
}

struct LruList {
    nodes: Vec<LruNode>,
    head: Option<usize>,
    tail: Option<usize>,
    free_indices: Vec<usize>,
}

impl LruList {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            head: None,
            tail: None,
            free_indices: Vec::new(),
        }
    }

    fn push_back(&mut self, dialogue_id: String) -> usize {
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

    fn remove(&mut self, index: usize) {
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

    fn pop_back(&mut self) -> Option<String> {
        let tail = self.tail?;
        let dialogue_id = self.nodes[tail].dialogue_id.clone();
        self.remove(tail);
        Some(dialogue_id)
    }

    fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    fn len(&self) -> usize {
        self.nodes.len() - self.free_indices.len()
    }
}

pub struct LruCacheStrategy {
    entries: RwLock<std::collections::HashMap<String, DialogueEntry>>,
    lru_list: Mutex<LruList>,
    retention_duration: Duration,
    max_entries: usize,
    slot_manager: Arc<SlotManager>,
}

impl LruCacheStrategy {
    pub fn new(
        slot_manager: Arc<SlotManager>,
        retention_duration: Duration,
        max_entries: usize,
    ) -> Self {
        Self {
            entries: RwLock::new(std::collections::HashMap::new()),
            lru_list: Mutex::new(LruList::new()),
            retention_duration,
            max_entries,
            slot_manager,
        }
    }

    pub async fn get(&self, dialogue_id: &str) -> Option<DialogueEntry> {
        let entries_read = self.entries.read().await;
        let entry = entries_read.get(dialogue_id)?;
        let entry_clone = entry.clone();
        drop(entries_read);

        let mut entries_write = self.entries.write().await;
        let entry = entries_write.get_mut(dialogue_id)?;

        if entry.in_lru {
            let mut lru_list = self.lru_list.lock().await;
            if let Some(lru_index) = entry.lru_index {
                lru_list.remove(lru_index);
            }
            entry.lru_index = None;
            entry.in_lru = false;
        }

        entry.last_accessed_at = Instant::now();

        Some(entry_clone)
    }

    pub async fn insert(&self, dialogue_id: String, slot_index: usize, token_count: usize) {
        let mut entries = self.entries.write().await;

        let entry = DialogueEntry {
            dialogue_id: dialogue_id.clone(),
            slot_index,
            token_count,
            last_accessed_at: Instant::now(),
            in_lru: false,
            lru_index: None,
        };

        entries.insert(dialogue_id, entry);

        drop(entries);

        self.cleanup(Instant::now()).await;
    }

    pub async fn remove(&self, dialogue_id: &str) {
        let mut entries = self.entries.write().await;

        if let Some(entry) = entries.remove(dialogue_id) {
            if entry.in_lru {
                let mut lru_list = self.lru_list.lock().await;
                if let Some(lru_index) = entry.lru_index {
                    lru_list.remove(lru_index);
                }
            }

            drop(entries);

            self.slot_manager.release_by_dialogue(dialogue_id).await;
        }
    }

    pub async fn cleanup(&self, now: Instant) {
        let to_evict: Vec<String> = {
            let mut entries = self.entries.write().await;
            let mut lru_list = self.lru_list.lock().await;

            for entry in entries.values_mut() {
                if !entry.in_lru && (now - entry.last_accessed_at) >= self.retention_duration {
                    entry.in_lru = true;
                    let lru_index = lru_list.push_back(entry.dialogue_id.clone());
                    entry.lru_index = Some(lru_index);
                }
            }

            let mut to_evict = Vec::new();
            while entries.len() > self.max_entries && !lru_list.is_empty() {
                if let Some(dialogue_id) = lru_list.pop_back() {
                    if let Some(entry) = entries.remove(&dialogue_id) {
                        to_evict.push(entry.dialogue_id);
                    }
                }
            }

            to_evict
        };

        for dialogue_id in to_evict {
            self.slot_manager.release_by_dialogue(&dialogue_id).await;
        }
    }

    pub async fn entry_count(&self) -> usize {
        self.entries.read().await.len()
    }

    pub async fn insert_batch(&self, entries: Vec<(String, usize, usize)>) {
        let mut lock = self.entries.write().await;
        for (dialogue_id, slot_index, token_count) in entries {
            lock.insert(
                dialogue_id.clone(),
                DialogueEntry {
                    dialogue_id,
                    slot_index,
                    token_count,
                    last_accessed_at: Instant::now(),
                    in_lru: false,
                    lru_index: None,
                },
            );
        }
        drop(lock);
        self.cleanup(Instant::now()).await;
    }

    pub async fn remove_batch(&self, dialogue_ids: &[&str]) {
        let mut to_release: Vec<String> = Vec::new();

        {
            let mut entries = self.entries.write().await;
            let mut lru_list = self.lru_list.lock().await;

            for &dialogue_id in dialogue_ids {
                if let Some(entry) = entries.remove(dialogue_id) {
                    if entry.in_lru {
                        if let Some(lru_index) = entry.lru_index {
                            lru_list.remove(lru_index);
                        }
                    }
                    to_release.push(entry.dialogue_id);
                }
            }
        }

        for dialogue_id in to_release {
            self.slot_manager.release_by_dialogue(&dialogue_id).await;
        }
    }
}
