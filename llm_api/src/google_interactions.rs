use serde::{Deserialize, Serialize};
use agent_models::response_item::{ContentPart, ResponseItem, Role};
use std::collections::HashMap;

// Data structures for Google's /v1/interactions API

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GeminiInteractionRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_interaction_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    pub role: String, // "user" or "model"
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
#[serde(untagged)]
pub enum Part {
    Text { text: String },
    InlineData { inline_data: InlineData },
    FunctionCall { function_call: FunctionCall },
    FunctionResponse { function_response: FunctionResponse },
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
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
                        // System role is mapped to user role prompt in Google Interactions format
                        Role::System => "user".to_string(), 
                        Role::Tool => continue, // Handled by FunctionCallOutput
                    };

                    let parts = content
                        .iter()
                        .map(|part| match part {
                            ContentPart::Text { text } => Part::Text { text: text.clone() },
                            ContentPart::Image { media_type, data_base64 } => Part::InlineData {
                                inline_data: InlineData {
                                    mime_type: media_type.clone(),
                                    data: data_base64.clone(),
                                },
                            },
                        })
                        .collect();
                    
                    contents.push(Content { role: gemini_role, parts });
                }
                ResponseItem::FunctionCall { name, arguments, call_id, .. } => {
                    let args: serde_json::Value = serde_json::from_str(arguments)
                        .unwrap_or_else(|_| serde_json::json!({ "raw": arguments }));
                    let call_part = Part::FunctionCall {
                        function_call: FunctionCall {
                            name: name.clone(),
                            args,
                        },
                    };
                    function_calls.insert(call_id.clone(), name.clone());

                    if let Some(last_content) = contents.last_mut() {
                        if last_content.role == "model" {
                            last_content.parts.push(call_part);
                            continue;
                        }
                    }
                    contents.push(Content {
                        role: "model".to_string(),
                        parts: vec![call_part],
                    });
                }
                ResponseItem::FunctionCallOutput { call_id, output, .. } => {
                    let name = function_calls.get(call_id).cloned().unwrap_or_else(|| "unknown_function".to_string());
                    let response: serde_json::Value = serde_json::from_str(output)
                        .unwrap_or_else(|_| serde_json::json!({ "output": output }));
                    contents.push(Content {
                        role: "user".to_string(),
                        parts: vec![Part::FunctionResponse {
                            function_response: FunctionResponse {
                                name,
                                response,
                            },
                        }],
                    });
                }
                ResponseItem::Reasoning { .. } => {
                    // Internal agent reasoning trace is not sent directly to Gemini endpoint
                }
            }
        }

        Ok(GeminiInteractionRequest {
            contents,
            previous_interaction_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_models::response_item::{ContentPart, ResponseItem, Role};

    #[test]
    fn test_to_gemini_request_conversation_flow() {
        let history = vec![
            ResponseItem::Message {
                id: "msg_sys".to_string(),
                role: Role::System,
                content: vec![ContentPart::Text {
                    text: "You are a helpful coding assistant.".to_string(),
                }],
            },
            ResponseItem::Message {
                id: "msg_user".to_string(),
                role: Role::User,
                content: vec![ContentPart::Text {
                    text: "What is the weather in London?".to_string(),
                }],
            },
            ResponseItem::Reasoning {
                id: "reason_1".to_string(),
                thought_process: "User is asking about London weather, need to call weather tool.".to_string(),
                signature: None,
            },
            ResponseItem::FunctionCall {
                id: "fc_1".to_string(),
                call_id: "call_123".to_string(),
                name: "get_weather".to_string(),
                arguments: r#"{"city":"London"}"#.to_string(),
            },
            ResponseItem::FunctionCallOutput {
                id: "fco_1".to_string(),
                call_id: "call_123".to_string(),
                output: r#"{"temperature": 18, "condition": "Cloudy"}"#.to_string(),
                is_error: false,
            },
            ResponseItem::Message {
                id: "msg_asst".to_string(),
                role: Role::Assistant,
                content: vec![ContentPart::Text {
                    text: "The weather in London is 18°C and cloudy.".to_string(),
                }],
            },
        ];

        let req = GoogleInteractionsAdapter::to_gemini_request(&history, Some("inter_prev_999".to_string())).unwrap();

        assert_eq!(req.previous_interaction_id, Some("inter_prev_999".to_string()));
        // Expected contents: System -> "user", User -> "user", FunctionCall -> "model", FunctionCallOutput -> "user", Assistant -> "model"
        assert_eq!(req.contents.len(), 5);
        assert_eq!(req.contents[0].role, "user");
        assert_eq!(req.contents[1].role, "user");
        assert_eq!(req.contents[2].role, "model");
        assert_eq!(req.contents[3].role, "user");
        assert_eq!(req.contents[4].role, "model");
    }

    #[test]
    fn test_gemini_image_part_handling() {
        let history = vec![ResponseItem::Message {
            id: "msg_img".to_string(),
            role: Role::User,
            content: vec![
                ContentPart::Text { text: "Describe this:".to_string() },
                ContentPart::Image {
                    media_type: "image/png".to_string(),
                    data_base64: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==".to_string(),
                },
            ],
        }];

        let req = GoogleInteractionsAdapter::to_gemini_request(&history, None).unwrap();
        assert_eq!(req.contents.len(), 1);
        assert_eq!(req.contents[0].parts.len(), 2);
    }
}
