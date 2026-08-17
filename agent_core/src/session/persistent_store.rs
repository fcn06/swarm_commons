use std::sync::Arc;
use redb::{Database, TableDefinition, ReadableTable, ReadableDatabase};
use agent_models::response_item::ResponseItem;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::Mutex;
use crate::session::{Session, SessionStore, SessionStoreApi};

const SESSIONS_TABLE: TableDefinition<&str, Vec<u8>> = TableDefinition::new("sessions");
const RESPONSE_INDEX_TABLE: TableDefinition<&str, &str> = TableDefinition::new("response_index");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentSessionRecord {
    pub id: String,
    pub parent_response_id: Option<String>,
    pub items: Vec<ResponseItem>,
    pub metadata: HashMap<String, String>,
}

pub struct PersistentSessionStore {
    cache: SessionStore,
    db: Arc<Database>,
    write_lock: Mutex<()>,
}

impl PersistentSessionStore {
    /// Open or create a redb-backed persistent session store at the given path
    pub fn new(db_path: &str) -> anyhow::Result<Self> {
        // Ensure parent directory exists if specified
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let db = Arc::new(Database::create(db_path)?);

        // Ensure tables exist
        {
            let write_txn = db.begin_write()?;
            {
                let _ = write_txn.open_table(SESSIONS_TABLE)?;
                let _ = write_txn.open_table(RESPONSE_INDEX_TABLE)?;
            }
            write_txn.commit()?;
        }

        let cache = SessionStore::new();

        // Hydrate in-memory cache from database
        let read_txn = db.begin_read()?;
        if let Ok(table) = read_txn.open_table(SESSIONS_TABLE) {
            for item in table.iter()? {
                let (key, val) = item?;
                let session_id = key.value();
                if let Ok(record) = serde_json::from_slice::<PersistentSessionRecord>(&val.value()) {
                    let session = cache.get_or_create(session_id);
                    // Set items in cache
                    if let Ok(mut items_guard) = session.items.try_write() {
                        *items_guard = record.items.clone();
                    }

                    if let Some(parent_id) = &record.parent_response_id {
                        cache.response_to_session.insert(parent_id.clone(), session_id.to_string());
                    }
                    for item in &record.items {
                        let item_id = match item {
                            ResponseItem::Message { id, .. } => id.clone(),
                            ResponseItem::Reasoning { id, .. } => id.clone(),
                            ResponseItem::FunctionCall { id, .. } => id.clone(),
                            ResponseItem::FunctionCallOutput { id, .. } => id.clone(),
                        };
                        cache.response_to_session.insert(item_id, session_id.to_string());
                    }
                }
            }
        }

        Ok(Self {
            cache,
            db,
            write_lock: Mutex::new(()),
        })
    }

    async fn persist_session(&self, session_id: &str) -> anyhow::Result<()> {
        let _lock = self.write_lock.lock().await;
        let history = self.cache.get_history(session_id).await;
        let parent_id = self.cache.get_parent_response_id(session_id).await;

        let record = PersistentSessionRecord {
            id: session_id.to_string(),
            parent_response_id: parent_id.clone(),
            items: history.clone(),
            metadata: HashMap::new(),
        };

        let encoded = serde_json::to_vec(&record)?;

        let write_txn = self.db.begin_write()?;
        {
            let mut sess_table = write_txn.open_table(SESSIONS_TABLE)?;
            sess_table.insert(session_id, encoded)?;

            let mut resp_table = write_txn.open_table(RESPONSE_INDEX_TABLE)?;
            if let Some(parent) = parent_id {
                resp_table.insert(parent.as_str(), session_id)?;
            }
            for item in &history {
                let item_id = match item {
                    ResponseItem::Message { id, .. } => id.as_str(),
                    ResponseItem::Reasoning { id, .. } => id.as_str(),
                    ResponseItem::FunctionCall { id, .. } => id.as_str(),
                    ResponseItem::FunctionCallOutput { id, .. } => id.as_str(),
                };
                resp_table.insert(item_id, session_id)?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl SessionStoreApi for PersistentSessionStore {
    async fn resolve_session(&self, previous_response_id: Option<&str>) -> Session {
        let session = self.cache.resolve_session(previous_response_id).await;
        let _ = self.persist_session(&session.id).await;
        session
    }

    async fn append_items(&self, session_id: &str, items: &[ResponseItem]) -> Vec<ResponseItem> {
        let updated = self.cache.append_items(session_id, items).await;
        let _ = self.persist_session(session_id).await;
        updated
    }

    async fn get_history(&self, session_id: &str) -> Vec<ResponseItem> {
        self.cache.get_history(session_id).await
    }

    async fn set_parent_response_id(&self, session_id: &str, parent_response_id: String) {
        self.cache.set_parent_response_id(session_id, parent_response_id).await;
        let _ = self.persist_session(session_id).await;
    }

    async fn get_parent_response_id(&self, session_id: &str) -> Option<String> {
        self.cache.get_parent_response_id(session_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_models::response_item::{ContentPart, Role};

    #[tokio::test]
    async fn test_persistent_session_store_lifecycle() {
        let temp_dir = std::env::temp_dir().join(format!("swarm_test_redb_{}", uuid::Uuid::new_v4()));
        let db_path = temp_dir.join("test_session.redb");
        let db_path_str = db_path.to_str().unwrap();

        // 1. Create and write to store
        {
            let store = PersistentSessionStore::new(db_path_str).unwrap();
            let session = store.resolve_session(None).await;
            let msg = ResponseItem::Message {
                id: "msg_persist_1".to_string(),
                role: Role::User,
                content: vec![ContentPart::Text {
                    text: "Persistent greeting".to_string(),
                }],
            };
            store.append_items(&session.id, &[msg]).await;
            store.set_parent_response_id(&session.id, "resp_parent_100".to_string()).await;
        }

        // 2. Re-open and verify persistence
        {
            let store2 = PersistentSessionStore::new(db_path_str).unwrap();
            let resolved = store2.resolve_session(Some("resp_parent_100")).await;
            let history = store2.get_history(&resolved.id).await;
            assert_eq!(history.len(), 1);
            if let ResponseItem::Message { content, .. } = &history[0] {
                if let ContentPart::Text { text } = &content[0] {
                    assert_eq!(text, "Persistent greeting");
                } else {
                    panic!("Expected text part");
                }
            } else {
                panic!("Expected Message");
            }
        }

        let _ = std::fs::remove_dir_all(temp_dir);
    }
}
