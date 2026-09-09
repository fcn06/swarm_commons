use anyhow::Result;
use async_trait::async_trait;
use llm_api::chat::Message;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct ContextRequest {
    pub session_id: Option<String>,
    pub agent_name: Option<String>,
    pub current_input: String,
    pub metadata: HashMap<String, String>,
}

#[async_trait]
pub trait ContextProvider: Send + Sync {
    fn name(&self) -> &str;
    async fn provide_context(&self, req: &ContextRequest) -> Result<Vec<Message>>;
}
