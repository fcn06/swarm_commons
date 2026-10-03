use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use dashmap::DashMap;
use tokio::sync::RwLock;
use agent_models::response_item::ResponseItem;

/// Trait to resolve session key from A2A session ID or metadata.
///
/// Composite keys should use `_` as separator, never `:`.
pub trait SessionKeyResolver: Send + Sync {
    /// Return None to run the turn without persisted history (ephemeral).
    fn resolve(
        &self,
        a2a_session_id: Option<&str>,
        metadata: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Option<String>;
}

/// Default resolver: returns `a2a_session_id` when present, else None (ephemeral).
/// Fixes cross-tenant leak / shared "default_session".
#[derive(Debug, Default, Clone)]
pub struct A2aSessionKeyResolver;

impl SessionKeyResolver for A2aSessionKeyResolver {
    fn resolve(
        &self,
        a2a_session_id: Option<&str>,
        _metadata: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Option<String> {
        a2a_session_id.map(|s| s.to_string())
    }
}

/// TenantThreadResolver example/implementation: resolves `{tenant}_{thread}`.
/// Returns None if neither tenant nor thread is resolvable and no session_id is provided.
#[derive(Debug, Default, Clone)]
pub struct TenantThreadResolver;

impl SessionKeyResolver for TenantThreadResolver {
    fn resolve(
        &self,
        a2a_session_id: Option<&str>,
        metadata: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Option<String> {
        let req_ctx = agent_models::RequestContext::from_metadata(metadata);
        if let (Some(tenant), Some(thread)) = (req_ctx.tenant_id, req_ctx.thread_id) {
            Some(format!("{}_{}", tenant, thread))
        } else if let Some(sid) = a2a_session_id {
            Some(sid.to_string())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub parent_response_id: Option<String>,
    pub items: Arc<RwLock<Vec<ResponseItem>>>,
    pub metadata: HashMap<String, String>,
    pub last_accessed: Arc<RwLock<Instant>>,
}

#[derive(Debug, Clone)]
pub struct SessionStore {
    sessions: Arc<DashMap<String, Session>>,
    // Mapping from response_id / parent_response_id to session_id for fast lookup
    response_to_session: Arc<DashMap<String, String>>,
    max_history_items: Option<usize>,
    ttl: Option<Duration>,
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            response_to_session: Arc::new(DashMap::new()),
            max_history_items: None,
            ttl: None,
        }
    }

    pub fn with_limits(max_history_items: Option<usize>, ttl: Option<Duration>) -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            response_to_session: Arc::new(DashMap::new()),
            max_history_items,
            ttl,
        }
    }

    pub fn get_or_create(&self, session_id: &str) -> Session {
        let now = Instant::now();
        let entry = self
            .sessions
            .entry(session_id.to_string())
            .or_insert_with(|| Session {
                id: session_id.to_string(),
                parent_response_id: None,
                items: Arc::new(RwLock::new(Vec::new())),
                metadata: HashMap::new(),
                last_accessed: Arc::new(RwLock::new(now)),
            });
        let session = entry.value().clone();
        session
    }

    /// Resolve or create a session id based on an optional previous_response_id.
    pub async fn resolve_session(&self, previous_response_id: Option<&str>) -> Session {
        if let Some(prev_id) = previous_response_id {
            if let Some(session_id) = self.response_to_session.get(prev_id) {
                let s = self.get_or_create(session_id.value());
                *s.last_accessed.write().await = Instant::now();
                return s;
            }
            if self.sessions.contains_key(prev_id) {
                let s = self.get_or_create(prev_id);
                *s.last_accessed.write().await = Instant::now();
                return s;
            }
            let session = self.get_or_create(prev_id);
            *session.last_accessed.write().await = Instant::now();
            session
        } else {
            let new_session_id = uuid::Uuid::new_v4().to_string();
            self.get_or_create(&new_session_id)
        }
    }

    pub async fn append_items(&self, session_id: &str, new_items: &[ResponseItem]) -> Vec<ResponseItem> {
        let session = self.get_or_create(session_id);
        *session.last_accessed.write().await = Instant::now();

        let mut items = session.items.write().await;
        for item in new_items {
            let item_id = match item {
                ResponseItem::Message { id, .. } => id.clone(),
                ResponseItem::Reasoning { id, .. } => id.clone(),
                ResponseItem::FunctionCall { id, .. } => id.clone(),
                ResponseItem::FunctionCallOutput { id, .. } => id.clone(),
            };
            self.response_to_session.insert(item_id, session_id.to_string());
        }
        items.extend(new_items.iter().cloned());

        // Enforce max_history_items limit if set by trimming oldest items
        if let Some(limit) = self.max_history_items {
            if items.len() > limit {
                let excess = items.len() - limit;
                items.drain(0..excess);
            }
        }

        items.clone()
    }

    pub async fn get_history(&self, session_id: &str) -> Vec<ResponseItem> {
        if let Some(session) = self.sessions.get(session_id) {
            *session.last_accessed.write().await = Instant::now();
            let items = session.items.read().await;
            items.clone()
        } else {
            Vec::new()
        }
    }

    pub async fn set_parent_response_id(&self, session_id: &str, parent_response_id: String) {
        self.response_to_session.insert(parent_response_id.clone(), session_id.to_string());
        if let Some(mut session) = self.sessions.get_mut(session_id) {
            *session.last_accessed.write().await = Instant::now();
            session.parent_response_id = Some(parent_response_id);
        }
    }

    pub async fn get_parent_response_id(&self, session_id: &str) -> Option<String> {
        self.sessions
            .get(session_id)
            .and_then(|session| session.parent_response_id.clone())
    }

    /// Spawns a background task that evicts expired sessions based on TTL.
    pub fn spawn_eviction(&self, interval: Duration) -> tokio::task::JoinHandle<()> {
        let sessions = Arc::clone(&self.sessions);
        let resp_to_session = Arc::clone(&self.response_to_session);
        let ttl = self.ttl;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                let Some(ttl_duration) = ttl else {
                    continue;
                };

                let now = Instant::now();
                let mut expired_session_ids = Vec::new();

                for entry in sessions.iter() {
                    let last_accessed = *entry.value().last_accessed.read().await;
                    if now.duration_since(last_accessed) > ttl_duration {
                        expired_session_ids.push(entry.key().clone());
                    }
                }

                for sid in expired_session_ids {
                    sessions.remove(&sid);
                    // Clean up response_to_session mappings pointing to this session
                    resp_to_session.retain(|_, v| v != &sid);
                }
            }
        })
    }
}

pub mod persistent_store;
pub use persistent_store::PersistentSessionStore;

#[async_trait::async_trait]
pub trait SessionStoreApi: Send + Sync {
    async fn resolve_session(&self, previous_response_id: Option<&str>) -> Session;
    async fn append_items(&self, session_id: &str, items: &[ResponseItem]) -> Vec<ResponseItem>;
    async fn get_history(&self, session_id: &str) -> Vec<ResponseItem>;
    async fn set_parent_response_id(&self, session_id: &str, parent_response_id: String);
    async fn get_parent_response_id(&self, session_id: &str) -> Option<String>;
}

#[async_trait::async_trait]
impl SessionStoreApi for SessionStore {
    async fn resolve_session(&self, previous_response_id: Option<&str>) -> Session {
        self.resolve_session(previous_response_id).await
    }

    async fn append_items(&self, session_id: &str, items: &[ResponseItem]) -> Vec<ResponseItem> {
        self.append_items(session_id, items).await
    }

    async fn get_history(&self, session_id: &str) -> Vec<ResponseItem> {
        self.get_history(session_id).await
    }

    async fn set_parent_response_id(&self, session_id: &str, parent_response_id: String) {
        self.set_parent_response_id(session_id, parent_response_id).await;
    }

    async fn get_parent_response_id(&self, session_id: &str) -> Option<String> {
        self.get_parent_response_id(session_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_models::response_item::{ContentPart, Role};

    #[tokio::test]
    async fn test_session_store_get_or_create_and_append() {
        let store = SessionStore::new();
        let session_id = "test_session_1";

        let item1 = ResponseItem::Message {
            id: "msg_1".to_string(),
            role: Role::User,
            content: vec![ContentPart::Text {
                text: "Hello".to_string(),
            }],
        };

        let history = store.append_items(session_id, &[item1.clone()]).await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0], item1);

        let retrieved_history = store.get_history(session_id).await;
        assert_eq!(retrieved_history.len(), 1);
        assert_eq!(retrieved_history[0], item1);
    }

    #[tokio::test]
    async fn test_session_limits_and_trimming() {
        let store = SessionStore::with_limits(Some(2), None);
        let session_id = "test_trim";

        let item1 = ResponseItem::Message {
            id: "msg_1".to_string(),
            role: Role::User,
            content: vec![ContentPart::Text { text: "1".to_string() }],
        };
        let item2 = ResponseItem::Message {
            id: "msg_2".to_string(),
            role: Role::User,
            content: vec![ContentPart::Text { text: "2".to_string() }],
        };
        let item3 = ResponseItem::Message {
            id: "msg_3".to_string(),
            role: Role::User,
            content: vec![ContentPart::Text { text: "3".to_string() }],
        };

        store.append_items(session_id, &[item1, item2, item3]).await;
        let history = store.get_history(session_id).await;
        assert_eq!(history.len(), 2);
        if let ResponseItem::Message { content, .. } = &history[0] {
            if let ContentPart::Text { text } = &content[0] {
                assert_eq!(text, "2");
            }
        }
    }

    #[tokio::test]
    async fn test_ttl_eviction() {
        let store = SessionStore::with_limits(None, Some(Duration::from_millis(50)));
        let session_id = "test_evict";
        let item = ResponseItem::Message {
            id: "msg_evict".to_string(),
            role: Role::User,
            content: vec![ContentPart::Text { text: "bye".to_string() }],
        };
        store.append_items(session_id, &[item]).await;
        let handle = store.spawn_eviction(Duration::from_millis(20));

        tokio::time::sleep(Duration::from_millis(100)).await;
        // Eviction should have run
        assert!(!store.sessions.contains_key(session_id));
        assert!(!store.response_to_session.contains_key("msg_evict"));
        handle.abort();
    }
}
