use crate::session::SessionStore;
use anyhow::Result;
use llm_api::google_interactions::{GeminiInteractionRequest, GoogleInteractionsAdapter};
use agent_models::response_item::{ContentPart, ResponseItem, Role};
use std::sync::Arc;
use uuid::Uuid;

pub struct InteractionHandler {
    session_store: Arc<SessionStore>,
    // In a real scenario, this would be a client to an LLM API
    // For now, we'll just use the adapter directly.
}

impl InteractionHandler {
    pub fn new(session_store: Arc<SessionStore>) -> Self {
        Self { session_store }
    }

    pub async fn process_request(
        &self,
        session_id: &str,
        user_message: String,
    ) -> Result<GeminiInteractionRequest> {
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

        // 4. Get the parent_response_id for stateless backends
        let parent_response_id = self
            .session_store
            .get_parent_response_id(session_id).await;

        // 5. Convert the history to the Gemini request format
        let gemini_request = GoogleInteractionsAdapter::to_gemini_request(&history, parent_response_id)?;

        // In a real implementation, you would now send this request to the Google API
        // and process the response.

        Ok(gemini_request)
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
                .process_request(session_id, "Hello, Gemini!".to_string())
                .await
                .unwrap();

            assert!(request1.previous_interaction_id.is_none());
            assert_eq!(request1.contents.len(), 1);
            assert_eq!(request1.contents[0].role, "user");
            

            // Mock a response from Gemini and update history
            let assistant_response = ResponseItem::Message {
                id: Uuid::new_v4().to_string(),
                role: Role::Assistant,
                content: vec![ContentPart::Text { text: "Hello there!".to_string() }]
            };
            session_store.append_items(session_id, &[assistant_response]).await;
            session_store.set_parent_response_id(session_id, "prev_id_123".to_string()).await;

            // Second interaction
            let request2 = handler
                .process_request(session_id, "How are you?".to_string())
                .await
                .unwrap();
                
            assert_eq!(request2.previous_interaction_id, Some("prev_id_123".to_string()));
            assert_eq!(request2.contents.len(), 3);
            assert_eq!(request2.contents[0].role, "user");
            assert_eq!(request2.contents[1].role, "model");
            assert_eq!(request2.contents[2].role, "user");
        });
    }
}
