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
}
