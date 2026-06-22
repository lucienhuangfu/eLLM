use std::time::Instant;

/// 会话模式
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SessionMode {
    Reusable,
    NonReusable,
}

/// 对话会话
#[derive(Debug, Clone)]
pub struct DialogueSession {
    /// 会话 ID
    pub session_id: String,
    /// 槽位索引
    pub slot_index: usize,
    /// token 数量
    pub token_count: usize,
    /// 创建时间
    pub created_at: Instant,
    /// 最后访问时间
    pub last_accessed: Instant,
}

impl DialogueSession {
    /// 更新最后访问时间
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }
}

/// 会话句柄
#[derive(Debug, Clone)]
pub struct SessionHandle {
    /// 会话 ID
    pub session_id: String,
    /// 槽位索引
    pub slot_index: usize,
    /// 是否复用
    pub is_reused: bool,
}

impl SessionHandle {
    /// 创建新的会话句柄
    pub fn new(session_id: String, slot_index: usize) -> Self {
        Self {
            session_id,
            slot_index,
            is_reused: false,
        }
    }

    /// 创建复用的会话句柄
    pub fn reused(session_id: String, slot_index: usize) -> Self {
        Self {
            session_id,
            slot_index,
            is_reused: true,
        }
    }
}