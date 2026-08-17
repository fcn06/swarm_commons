use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value; // Import Value for flexible parameters

use tracing::{ debug,warn};

use crate::tools::Tool;
use anyhow::{Result,Context};

use tokio::time::{sleep, Duration};

use regex::Regex;


#[derive(Clone)]
pub struct ChatLlmInteraction {
    pub client: Client,
    pub llm_url:String,
    llm_api_key:String,
    pub model_id:String,

}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>, // Keep existing message structure for history

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    // --- Tool Calling Additions (Request) ---
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    // --- End Tool Calling Additions ---
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,    // "system", "user", "assistant", or "tool"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>, // Content for system/user/assistant, or result for tool

    // --- Tool Calling Additions (for Tool Result Message) ---
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>, // Only used when role is "tool"
    // --- End Tool Calling Additions ---

    // --- Add tool_calls to Message for assistant messages ---
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)] // Handles string ("none", "auto") or object variants
pub enum ToolChoice {
    String(String), // Represents "none" or "auto"
    Function {
        r#type: String, // Should be "function"
        function: FunctionName,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionName {
    pub name: String,
}

// --- Structs for Response ---

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage, // Use the modified message struct
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>, // Or specific struct if needed
    pub finish_reason: String,    // "stop", "length", "tool_calls", etc.
}

// --- Modified Response Message & Tool Call Structs (Response) ---
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResponseMessage {
    pub role: String, // "assistant"
    // Content might be null if tool_calls is present
    pub content: Option<String>,
    // Tool calls requested by the model
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize,Deserialize, Debug, Clone)]
pub struct ToolCall {
    pub id: String,     // ID to be sent back in the tool result message
    pub r#type: String, // Typically "function"
    pub function: FunctionCall,
}

#[derive(Serialize,Deserialize, Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    // Arguments is a STRING containing JSON, needs parsing
    pub arguments: String,
}
// --- End Tool Call Structs ---

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// --- API Call Function ---

impl ChatLlmInteraction {
    /// Create a new llm interaction entity
    pub fn new(llm_url:String,model_id:String,llm_api_key:String) -> Self {
        
        Self {
            client: reqwest::Client::builder().timeout(std::time::Duration::from_secs(30)).build().unwrap_or_else(|_| reqwest::Client::new()),
            llm_url:llm_url,
            llm_api_key:llm_api_key,
            model_id:model_id,
        }
    }

    /// Unified API call function for chat completions, handling both simple messages and tool calls.
    pub async fn call_api(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
    ) -> anyhow::Result<Option<Message>> {

        let llm_request_payload = ChatCompletionRequest {
            model: self.model_id.clone(),
            messages,
            temperature: Some(0.7),
            max_tokens: None,
            top_p: None,
            stop: None,
            stream: None,
            tools,
            tool_choice,
        };

        let llm_response = self.call_chat_completions_v2(&llm_request_payload)
            .await
            .context("LLM API request failed")?;

        let response_message = llm_response
            .choices
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("LLM response missing choices"))?
            .message
            .clone();


            let final_content = if let Some(content_str) = response_message.content.clone() {
                let cleaned_content_str = self.remove_think_tags(content_str).await?;
                
                // Attempt to parse content as JSON. If successful, handle Value::String separately
                if let Ok(json_value) = serde_json::from_str::<Value>(&cleaned_content_str) {
                    match json_value {
                        Value::String(s) => Some(s), // If it's a JSON string, take the inner string
                        _ => Some(serde_json::to_string(&json_value).context("Failed to re-serialize JSON content")?), // Otherwise, re-serialize
                    }
                } else {
                    Some(cleaned_content_str)
                }
            } else {
                None
            };

        debug!("Final Content: {}", final_content.clone().unwrap_or_default());
        

        Ok(Some(Message {
            role: response_message.role,
            content: final_content,
            tool_call_id: None, // This will be set on tool result messages, not assistant messages
            tool_calls: response_message.tool_calls,
        }))
    }



    /// for complex calls using tools, this is the api to use
    pub async fn call_chat_completions_v2(
        &self,
        request_payload: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, reqwest::Error> {
        
        let mut retries = 0;
        //let max_retries = 7; // You can adjust this
        //let mut delay = Duration::from_secs(2); // Starting delay
        let max_retries = 2;
        let mut delay = Duration::from_secs(2);
        let max_delay = Duration::from_secs(10); // Hard cap on any single wait


        loop {
            let response = self.client
                .post(self.llm_url.clone())
                .bearer_auth(self.llm_api_key.clone())
                .header("Content-Type", "application/json; charset=utf-8")
                .json(request_payload)
                .send()
                .await?;

            debug!("LLM API Response : {:?}", response);

            if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                retries += 1;
                if retries > max_retries {
                    return Err(response.error_for_status().unwrap_err()); // Return the last 429 error
                }
                
                // Check if the API tells us exactly how long to wait
                let retry_after = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|h| h.to_str().ok())
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(Duration::from_secs_f64);
                //let wait_time = retry_after.unwrap_or(delay);
                
                // Cap Retry-After: Groq often sends 500-900s which blocks the agent
                let wait_time = retry_after
                    .map(|ra| ra.min(max_delay))
                    .unwrap_or(delay)
                    .min(max_delay);
                
                warn!("Rate limit hit (429). Retrying in {:?}... (Retry {}/{})", wait_time, retries, max_retries);
                sleep(wait_time).await;
                
                if retry_after.is_none() {
                    // delay *= 2; // Exponential backoff only if no explicit Retry-After
                    delay = (delay * 2).min(max_delay);
                }
            } else {
                // Check for other HTTP errors and log the raw body before failing
                if let Err(e) = response.error_for_status_ref() {
                    let err_body = response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
                    tracing::error!("LLM API returned HTTP {}: {}", e.status().map(|s| s.as_u16()).unwrap_or(0), err_body);
                    return Err(e);
                }
                
                let response_body = response.json::<ChatCompletionResponse>().await?;
                debug!("LLM API Response Body: {:?}", response_body);
                return Ok(response_body);
            }
        }
    }


    /// Stream chat completions from the upstream LLM, emitting parsed delta chunks
    pub async fn call_chat_completions_stream(
        &self,
        request_payload: &ChatCompletionRequest,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<Option<Usage>, reqwest::Error> {
        let mut stream_payload = request_payload.clone();
        stream_payload.stream = Some(true);

        let response = self.client
            .post(self.llm_url.clone())
            .bearer_auth(self.llm_api_key.clone())
            .header("Content-Type", "application/json; charset=utf-8")
            .json(&stream_payload)
            .send()
            .await?;

        if let Err(e) = response.error_for_status_ref() {
            let err_body = response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
            tracing::error!("LLM streaming API returned HTTP {}: {}", e.status().map(|s| s.as_u16()).unwrap_or(0), err_body);
            return Err(e);
        }

        let mut response = response;
        let mut buffer = String::new();
        let mut usage: Option<Usage> = None;

        while let Some(chunk) = response.chunk().await? {
            let chunk_str = String::from_utf8_lossy(&chunk);
            buffer.push_str(&chunk_str);

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer.drain(..=pos);

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data:") {
                    let data = data.trim();
                    if data == "[DONE]" {
                        let _ = tx.send("[DONE]".to_string()).await;
                        break;
                    }
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(u) = val.get("usage").and_then(|u| if u.is_null() { None } else { Some(u) }) {
                            if let Ok(parsed_usage) = serde_json::from_value::<Usage>(u.clone()) {
                                usage = Some(parsed_usage);
                            }
                        }
                    }
                    let _ = tx.send(data.to_string()).await;
                }
            }
        }

        Ok(usage)
    }

    // Helper function to extract text from a Message using precompiled regexes
    pub async fn remove_think_tags(&self, result: String) -> anyhow::Result<String> {
        lazy_static::lazy_static! {
            static ref RE_JSON: Regex = Regex::new(r"```json\s*([\s\S]*?)\s*```").unwrap();
            static ref RE_CODE: Regex = Regex::new(r"```\s*([\s\S]*?)\s*```").unwrap();
            static ref RE_THINK: Regex = Regex::new(r"<think>([\s\S]*?)</think>").unwrap();
        }

        let mut cleaned = RE_JSON.replace_all(&result, "$1").to_string();
        cleaned = RE_CODE.replace_all(&cleaned, "$1").to_string();
        cleaned = RE_THINK.replace_all(&cleaned, "").to_string();

        Ok(cleaned.trim().to_string())
    }


    /********************************************/
    // Below functions should be deprecated in future
    /********************************************/

    /* 
    /// for very simple calls without tools, one can use this simpler api. Returns an Option<Message> for LLM
    pub async fn call_api_message(
        &self,
        messages: Vec<Message>,
    ) -> anyhow::Result<Option<Message>> {
        self.call_api(messages, None, None).await
    }
    */

    /// for very simple calls without tools, one can use this simpler api. This api returns a String from llm instead of Option<Message>
    /// This one should be used preferabbly
    pub async fn call_api_simple_v2(
        &self,
        agent_role: String,
        user_query: String,
    ) -> anyhow::Result<Option<String>> {
        let messages = vec![Message {
            role: agent_role,
            content: Some(user_query),
            tool_call_id: None,
            tool_calls: None,
        }];

        let response_message = self.call_api(messages, None, None).await?;
        Ok(response_message.and_then(|msg| msg.content))
    }

    pub async fn call_api_simple(
        &self,
        agent_role: String,
        user_query: String,
    ) -> anyhow::Result<Option<Message>> {

        self.call_api_simple_v2(agent_role, user_query).await.and_then(|s| {
            Ok(s.map(|content| Message {
                role: "assistant".to_string(),
                content: Some(content),
                tool_call_id: None,
                tool_calls: None,
            }))
        })
    }
}
