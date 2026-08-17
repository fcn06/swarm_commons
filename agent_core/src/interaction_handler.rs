use crate::session::SessionStore;
use anyhow::Result;
use agent_models::agent_request::AgentRequest;
use agent_models::response_item::{ContentPart, ResponseItem, Role};
use std::sync::Arc;
use uuid::Uuid;

pub struct InteractionHandler {
    session_store: Arc<SessionStore>,
}

impl InteractionHandler {
    pub fn new(session_store: Arc<SessionStore>) -> Self {
        Self { session_store }
    }

    pub async fn process_request(
        &self,
        session_id: &str,
        user_message: String,
    ) -> Result<AgentRequest> {
        // 1. Create a new ResponseItem for the user's message
        let user_item = ResponseItem::Message {
            id: Uuid::new_v4().to_string(),
            role: Role::User,
            content: vec![ContentPart::Text { text: user_message }],
        };

        // 2. Append the new item to the session history
        self.session_store.append_items(session_id, &[user_item]).await;

        // 3. Get the full, updated history
        let history = self.session_store.get_history(session_id).await;

        // 4. Return provider-agnostic AgentRequest
        Ok(AgentRequest {
            items: history,
            session_id: Some(session_id.to_string()),
            metadata: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;

    #[test]
    fn test_process_request_conversion() {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let session_store = Arc::new(SessionStore::new());
            let handler = InteractionHandler::new(session_store.clone());
            let session_id = "test_session_interactions";

            // First interaction
            let request1 = handler
                .process_request(session_id, "Hello, Agent!".to_string())
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
                .process_request(session_id, "How are you?".to_string())
                .await
                .unwrap();

            assert_eq!(request2.items.len(), 3);
            assert_eq!(request2.user_query(), "How are you?");
        });
    }
}
