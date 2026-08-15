use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseItem {
    /// Standard conversational turn
    Message {
        id: String,
        role: Role,
        content: Vec<ContentPart>,
    },
    /// Agent reasoning / chain-of-thought trace
    Reasoning {
        id: String,
        thought_process: String,
        signature: Option<String>,
    },
    /// Model invocation of a local or remote tool
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String, // JSON serialized string
    },
    /// Execution output resulting from a tool invocation
    FunctionCallOutput {
        id: String,
        call_id: String,
        output: String,
        is_error: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image { media_type: String, data_base64: String },
}

/// Open Responses: Input format can be a single prompt string or a list of ResponseItems
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponseItem>),
}

/// Open Responses: Request schema for POST /v1/responses
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CreateResponseRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub input: Option<ResponsesInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
}

/// Open Responses: Response schema for POST /v1/responses (non-streaming)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub output: Vec<ResponseItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serde() {
        let item = ResponseItem::Message {
            id: "msg_1".to_string(),
            role: Role::User,
            content: vec![ContentPart::Text {
                text: "Hello, swarm!".to_string(),
            }],
        };

        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"type\":\"message\""));
        assert!(json.contains("\"role\":\"user\""));

        let deserialized: ResponseItem = serde_json::from_str(&json).unwrap();
        assert_eq!(item, deserialized);
    }

    #[test]
    fn test_reasoning_serde() {
        let item = ResponseItem::Reasoning {
            id: "reason_1".to_string(),
            thought_process: "Analyzing user input...".to_string(),
            signature: Some("sig_123".to_string()),
        };

        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"type\":\"reasoning\""));

        let deserialized: ResponseItem = serde_json::from_str(&json).unwrap();
        assert_eq!(item, deserialized);
    }

    #[test]
    fn test_function_call_and_output_serde() {
        let call = ResponseItem::FunctionCall {
            id: "fc_1".to_string(),
            call_id: "call_abc".to_string(),
            name: "get_weather".to_string(),
            arguments: "{\"city\":\"Paris\"}".to_string(),
        };

        let output = ResponseItem::FunctionCallOutput {
            id: "fco_1".to_string(),
            call_id: "call_abc".to_string(),
            output: "{\"temp\": 22}".to_string(),
            is_error: false,
        };

        let json_call = serde_json::to_string(&call).unwrap();
        let json_output = serde_json::to_string(&output).unwrap();

        assert_eq!(call, serde_json::from_str(&json_call).unwrap());
        assert_eq!(output, serde_json::from_str(&json_output).unwrap());
    }

    #[test]
    fn test_open_responses_request_serde() {
        let req_str = r#"{
            "model": "gpt-4o",
            "input": "What is the capital of France?",
            "previous_response_id": "resp_prev_123",
            "stream": false
        }"#;

        let req: CreateResponseRequest = serde_json::from_str(req_str).unwrap();
        assert_eq!(req.model.as_deref(), Some("gpt-4o"));
        assert_eq!(req.previous_response_id.as_deref(), Some("resp_prev_123"));
        match req.input {
            Some(ResponsesInput::Text(s)) => assert_eq!(s, "What is the capital of France?"),
            _ => panic!("Expected text input"),
        }

        let items_req_str = r#"{
            "input": [
                {
                    "type": "message",
                    "id": "m1",
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}]
                }
            ]
        }"#;

        let req2: CreateResponseRequest = serde_json::from_str(items_req_str).unwrap();
        match req2.input {
            Some(ResponsesInput::Items(items)) => {
                assert_eq!(items.len(), 1);
            }
            _ => panic!("Expected items input"),
        }
    }
}
