use crate::session::SessionStoreApi;
use anyhow::Result;
use agent_models::agent_request::AgentRequest;
use agent_models::response_item::{ContentPart, ResponseItem, Role};
use std::sync::Arc;
use uuid::Uuid;

pub struct InteractionHandler {
    session_store: Arc<dyn SessionStoreApi>,
}

impl InteractionHandler {
    pub fn new(session_store: Arc<dyn SessionStoreApi>) -> Self {
        Self { session_store }
    }

    pub async fn process_request(
        &self,
        session_id: Option<&str>,
        user_message: String,
    ) -> Result<AgentRequest> {
        // 1. Create a new ResponseItem for the user's message
        let user_item = ResponseItem::Message {
            id: Uuid::new_v4().to_string(),
            role: Role::User,
            content: vec![ContentPart::Text { text: user_message }],
        };

        match session_id {
            Some(sid) => {
                // Persistent / stateful turn
                self.session_store.append_items(sid, &[user_item]).await;
                let history = self.session_store.get_history(sid).await;
                Ok(AgentRequest {
                    items: history,
                    session_id: Some(sid.to_string()),
                    metadata: None,
                })
            }
            None => {
                // Ephemeral turn: no session persistence
                Ok(AgentRequest {
                    items: vec![user_item],
                    session_id: None,
                    metadata: None,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionStore;
    use tokio::runtime::Runtime;

    #[test]
    fn test_process_request_conversion() {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let session_store: Arc<dyn SessionStoreApi> = Arc::new(SessionStore::new());
            let handler = InteractionHandler::new(session_store.clone());
            let session_id = "test_session_interactions";

            // First interaction
            let request1 = handler
                .process_request(Some(session_id), "Hello, Agent!".to_string())
                .await
                .unwrap();

            assert_eq!(request1.items.len(), 1);
            assert_eq!(request1.user_query(), "Hello, Agent!");

            // Mock a response and update history
            let assistant_response = ResponseItem::Message {
                id: Uuid::new_v4().to_string(),
                role: Role::Assistant,
                content: vec![ContentPart::Text { text: "Hello there!".to_string() }]
            };
            session_store.append_items(session_id, &[assistant_response]).await;

            // Second interaction
            let request2 = handler
                .process_request(Some(session_id), "How are you?".to_string())
                .await
                .unwrap();

            assert_eq!(request2.items.len(), 3);
            assert_eq!(request2.user_query(), "How are you?");

            // Ephemeral interaction
            let ephemeral_req = handler
                .process_request(None, "Ephemeral message".to_string())
                .await
                .unwrap();
            assert_eq!(ephemeral_req.items.len(), 1);
            assert_eq!(ephemeral_req.session_id, None);
        });
    }
}
