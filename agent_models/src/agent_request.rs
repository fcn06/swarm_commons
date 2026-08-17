use crate::response_item::{ContentPart, ResponseItem, Role};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Provider-agnostic request passed to an Agent's handle_request method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentRequest {
    /// Conversation history items
    pub items: Vec<ResponseItem>,
    /// Optional session ID
    pub session_id: Option<String>,
    /// Optional metadata
    pub metadata: Option<Map<String, Value>>,
}

impl AgentRequest {
    pub fn new(items: Vec<ResponseItem>) -> Self {
        Self {
            items,
            session_id: None,
            metadata: None,
        }
    }

    pub fn from_user_query(text: impl Into<String>) -> Self {
        let msg = ResponseItem::Message {
            id: format!("msg_{}", uuid::Uuid::new_v4()),
            role: Role::User,
            content: vec![ContentPart::Text { text: text.into() }],
        };
        Self::new(vec![msg])
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_metadata(mut self, metadata: Option<Map<String, Value>>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Helper to extract the latest user message text from items
    pub fn user_query(&self) -> String {
        self.items
            .iter()
            .rev()
            .find_map(|item| match item {
                ResponseItem::Message { role: Role::User, content, .. } => {
                    let text = content
                        .iter()
                        .filter_map(|part| match part {
                            ContentPart::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !text.is_empty() {
                        Some(text)
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .unwrap_or_default()
    }
}
