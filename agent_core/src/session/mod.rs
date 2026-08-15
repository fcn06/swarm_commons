use std::collections::HashMap;
use std::sync::Arc;
use dashmap::DashMap;
use tokio::sync::RwLock;
use agent_models::response_item::ResponseItem;

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub parent_response_id: Option<String>,
    pub items: Arc<RwLock<Vec<ResponseItem>>>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Default)]
pub struct SessionStore {
    sessions: DashMap<String, Session>,
    // Mapping from response_id / parent_response_id to session_id for fast lookup
    response_to_session: DashMap<String, String>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
            response_to_session: DashMap::new(),
        }
    }

    pub async fn get_or_create(&self, session_id: &str) -> Session {
        self.sessions
            .entry(session_id.to_string())
            .or_insert_with(|| Session {
                id: session_id.to_string(),
                parent_response_id: None,
                items: Arc::new(RwLock::new(Vec::new())),
                metadata: HashMap::new(),
            })
            .value()
            .clone()
    }

    /// Resolve or create a session id based on an optional previous_response_id.
    /// If previous_response_id matches an existing session ID or registered response ID, that session is used.
    /// Otherwise, if previous_response_id is provided, it is used as a new session ID or mapped.
    pub async fn resolve_session(&self, previous_response_id: Option<&str>) -> Session {
        if let Some(prev_id) = previous_response_id {
            if let Some(session_id) = self.response_to_session.get(prev_id) {
                return self.get_or_create(session_id.value()).await;
            }
            if self.sessions.contains_key(prev_id) {
                return self.get_or_create(prev_id).await;
            }
            // Create a new session with id matching or referencing previous_response_id
            let session = self.get_or_create(prev_id).await;
            session
        } else {
            let new_session_id = uuid::Uuid::new_v4().to_string();
            self.get_or_create(&new_session_id).await
        }
    }

    pub async fn append_items(&self, session_id: &str, new_items: &[ResponseItem]) -> Vec<ResponseItem> {
        let session = self.get_or_create(session_id).await;
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
        items.clone()
    }

    pub async fn get_history(&self, session_id: &str) -> Vec<ResponseItem> {
        if let Some(session) = self.sessions.get(session_id) {
            let items = session.items.read().await;
            items.clone()
        } else {
            Vec::new()
        }
    }

    pub async fn set_parent_response_id(&self, session_id: &str, parent_response_id: String) {
        self.response_to_session.insert(parent_response_id.clone(), session_id.to_string());
        if let Some(mut session) = self.sessions.get_mut(session_id) {
            session.parent_response_id = Some(parent_response_id);
        }
    }

    pub async fn get_parent_response_id(&self, session_id: &str) -> Option<String> {
        self.sessions
            .get(session_id)
            .and_then(|session| session.parent_response_id.clone())
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
    async fn test_session_resolution_by_previous_response_id() {
        let store = SessionStore::new();
        let s1 = store.resolve_session(None).await;
        let item1 = ResponseItem::Message {
            id: "resp_msg_100".to_string(),
            role: Role::Assistant,
            content: vec![ContentPart::Text {
                text: "Turn 1 answer".to_string(),
            }],
        };
        store.append_items(&s1.id, &[item1]).await;
        store.set_parent_response_id(&s1.id, "resp_msg_100".to_string()).await;

        // Next request provides previous_response_id
        let s2 = store.resolve_session(Some("resp_msg_100")).await;
        assert_eq!(s1.id, s2.id);

        let history = store.get_history(&s2.id).await;
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn test_concurrent_session_store() {
        let store = Arc::new(SessionStore::new());
        let session_id = "concurrent_session";

        let mut handles = vec![];
        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = tokio::spawn(async move {
                let item = ResponseItem::Message {
                    id: format!("msg_{i}"),
                    role: Role::User,
                    content: vec![ContentPart::Text {
                        text: format!("Message {i}"),
                    }],
                };
                store_clone.append_items(session_id, &[item]).await;
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }
        let history = store.get_history(session_id).await;
        assert_eq!(history.len(), 10);
    }
}
