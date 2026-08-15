use serde::{Deserialize, Serialize};
use agent_models::response_item::{ContentPart, ResponseItem, Role};
use std::collections::HashMap;

// Data structures for Google's /v1/interactions API

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiInteractionRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_interaction_id: Option<String>,
    // Add other fields like tools, safetySettings etc. as needed
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    pub role: String, // "user" or "model"
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
#[serde(untagged)]
pub enum Part {
    Text { text: String },
    FunctionCall { function_call: FunctionCall },
    FunctionResponse { function_response: FunctionResponse },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

// --- Adapter Logic ---

pub struct GoogleInteractionsAdapter;

impl GoogleInteractionsAdapter {
    /// Converts a history of `ResponseItem`s into a `GeminiInteractionRequest`.
    pub fn to_gemini_request(
        history: &[ResponseItem],
        previous_interaction_id: Option<String>,
    ) -> Result<GeminiInteractionRequest, serde_json::Error> {
        let mut contents = Vec::new();
        let mut function_calls = HashMap::new();

        for item in history {
            match item {
                ResponseItem::Message { role, content, .. } => {
                    let gemini_role = match role {
                        Role::User => "user".to_string(),
                        Role::Assistant => "model".to_string(),
                        // System role needs to be handled, often as the first user message
                        Role::System => "user".to_string(), 
                        Role::Tool => continue, // Handled by FunctionCallOutput
                    };

                    let parts = content
                        .iter()
                        .map(|part| match part {
                            ContentPart::Text { text } => Part::Text { text: text.clone() },
                            // Image conversion would go here
                            _ => unimplemented!(),
                        })
                        .collect();
                    
                    contents.push(Content { role: gemini_role, parts });
                }
                ResponseItem::FunctionCall { name, arguments, call_id, .. } => {
                     // This assumes a function call is always preceded by a model message
                    if let Some(last_content) = contents.last_mut() {
                        if last_content.role == "model" {
                            let args: serde_json::Value = serde_json::from_str(arguments)?;
                            last_content.parts.push(Part::FunctionCall {
                                function_call: FunctionCall {
                                    name: name.clone(),
                                    args,
                                },
                            });
                            function_calls.insert(call_id.clone(), name.clone());
                        }
                    }
                }
                ResponseItem::FunctionCallOutput { call_id, output, .. } => {
                    if let Some(name) = function_calls.get(call_id) {
                        let response: serde_json::Value = serde_json::from_str(output)?;
                        contents.push(Content {
                            role: "user".to_string(), // In Gemini, function responses are in a user role content
                            parts: vec![Part::FunctionResponse {
                                function_response: FunctionResponse {
                                    name: name.clone(),
                                    response,
                                },
                            }],
                        });
                    }
                }
                ResponseItem::Reasoning { .. } => {
                    // Reasoning is internal to the swarm, not sent to Gemini
                }
            }
        }

        Ok(GeminiInteractionRequest {
            contents,
            previous_interaction_id,
        })
    }
}
