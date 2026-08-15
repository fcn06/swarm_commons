use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::post,
    Json, Router,
};
use futures::stream;
use serde::Serialize;
use uuid::Uuid;

use agent_models::response_item::{
    ContentPart, CreateResponseRequest, ResponseItem, ResponseObject, ResponseUsage, ResponsesInput, Role,
};
use llm_api::chat::{
    ChatCompletionRequest, ChatCompletionResponse, Choice, ResponseMessage, Usage,
};

use crate::session::SessionStore;

/// Trait for handling the gateway generation backend (e.g. LLM call, agent orchestration loop)
#[async_trait::async_trait]
pub trait GatewayBackend: Send + Sync {
    async fn process_turn(
        &self,
        session_id: &str,
        history: &[ResponseItem],
        model: Option<&str>,
    ) -> Result<Vec<ResponseItem>, String>;
}

/// A default Echo/Mock backend or forwarding backend for the gateway
pub struct SimpleGatewayBackend;

#[async_trait::async_trait]
impl GatewayBackend for SimpleGatewayBackend {
    async fn process_turn(
        &self,
        _session_id: &str,
        history: &[ResponseItem],
        _model: Option<&str>,
    ) -> Result<Vec<ResponseItem>, String> {
        let last_user_text = history
            .iter()
            .rev()
            .find_map(|item| match item {
                ResponseItem::Message { role: Role::User, content, .. } => {
                    content.iter().find_map(|p| match p {
                        ContentPart::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                }
                _ => None,
            })
            .unwrap_or_else(|| "Echo from Swarm Gateway".to_string());

        let response_item = ResponseItem::Message {
            id: format!("resp_msg_{}", Uuid::new_v4()),
            role: Role::Assistant,
            content: vec![ContentPart::Text {
                text: format!("Swarm Gateway Response: {}", last_user_text),
            }],
        };

        Ok(vec![response_item])
    }
}

/// Multi-model gateway backend capable of routing to Google Gemini, Groq, OpenAI, or local models
pub struct MultiModelGatewayBackend {
    pub client: reqwest::Client,
    pub gemini_api_key: Option<String>,
    pub groq_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub custom_endpoint: Option<String>,
}

impl Default for MultiModelGatewayBackend {
    fn default() -> Self {
        Self::from_env()
    }
}

impl MultiModelGatewayBackend {
    pub fn from_env() -> Self {
        let gemini_api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("LLM_GEMINI_API_KEY"))
            .ok();
        let groq_api_key = std::env::var("GROQ_API_KEY")
            .or_else(|_| std::env::var("LLM_GROQ_API_KEY"))
            .or_else(|_| std::env::var("LLM_API_KEY"))
            .ok();
        let openai_api_key = std::env::var("OPENAI_API_KEY").ok();
        let custom_endpoint = std::env::var("SWARM_LLM_URL").ok();

        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(45))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            gemini_api_key,
            groq_api_key,
            openai_api_key,
            custom_endpoint,
        }
    }
}

#[async_trait::async_trait]
impl GatewayBackend for MultiModelGatewayBackend {
    async fn process_turn(
        &self,
        session_id: &str,
        history: &[ResponseItem],
        model: Option<&str>,
    ) -> Result<Vec<ResponseItem>, String> {
        let model_str = model.unwrap_or("gemini-2.0-flash");

        // 1. Google Gemini routing via GoogleInteractionsAdapter
        if (model_str.contains("gemini") || model_str.starts_with("google")) && self.gemini_api_key.is_some() {
            let key = self.gemini_api_key.as_ref().unwrap();
            let gemini_req = llm_api::google_interactions::GoogleInteractionsAdapter::to_gemini_request(
                history,
                Some(session_id.to_string()),
            ).map_err(|e| format!("Failed to build Gemini interaction request: {}", e))?;

            let target_model = model_str.strip_prefix("google/").unwrap_or(model_str);
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                target_model, key
            );

            let res = self.client.post(&url)
                .json(&gemini_req)
                .send()
                .await
                .map_err(|e| format!("Gemini API request failed: {}", e))?;

            if !res.status().is_success() {
                let err_text = res.text().await.unwrap_or_default();
                return Err(format!("Gemini API error: {}", err_text));
            }

            #[derive(serde::Deserialize)]
            struct GeminiResponse {
                candidates: Option<Vec<GeminiCandidate>>,
            }
            #[derive(serde::Deserialize)]
            struct GeminiCandidate {
                content: Option<GeminiContent>,
            }
            #[derive(serde::Deserialize)]
            struct GeminiContent {
                parts: Option<Vec<GeminiPartResponse>>,
            }
            #[derive(serde::Deserialize)]
            #[serde(untagged)]
            enum GeminiPartResponse {
                Text { text: String },
                FunctionCall { function_call: serde_json::Value },
            }

            let gemini_data: GeminiResponse = res.json().await
                .map_err(|e| format!("Failed to parse Gemini response: {}", e))?;

            let mut output_items = Vec::new();
            if let Some(candidates) = gemini_data.candidates {
                if let Some(first) = candidates.into_iter().next() {
                    if let Some(content) = first.content {
                        if let Some(parts) = content.parts {
                            for part in parts {
                                match part {
                                    GeminiPartResponse::Text { text } => {
                                        output_items.push(ResponseItem::Message {
                                            id: format!("resp_msg_{}", Uuid::new_v4()),
                                            role: Role::Assistant,
                                            content: vec![ContentPart::Text { text }],
                                        });
                                    }
                                    GeminiPartResponse::FunctionCall { function_call } => {
                                        let name = function_call.get("name").and_then(|v| v.as_str()).unwrap_or("unknown_tool").to_string();
                                        let args = function_call.get("args").map(|v| v.to_string()).unwrap_or_default();
                                        output_items.push(ResponseItem::FunctionCall {
                                            id: format!("fc_{}", Uuid::new_v4()),
                                            call_id: format!("call_{}", Uuid::new_v4()),
                                            name,
                                            arguments: args,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !output_items.is_empty() {
                return Ok(output_items);
            }
        }

        // 2. Groq / OpenAI / Custom chat completion routing
        let (endpoint, api_key) = if let Some(custom) = &self.custom_endpoint {
            (custom.clone(), self.openai_api_key.clone().unwrap_or_default())
        } else if model_str.starts_with("groq/") || self.groq_api_key.is_some() {
            (
                "https://api.groq.com/openai/v1/chat/completions".to_string(),
                self.groq_api_key.clone().unwrap_or_default(),
            )
        } else if let Some(key) = &self.openai_api_key {
            (
                "https://api.openai.com/v1/chat/completions".to_string(),
                key.clone(),
            )
        } else {
            // Fallback to SimpleGatewayBackend echo
            return SimpleGatewayBackend.process_turn(session_id, history, model).await;
        };

        if !api_key.is_empty() {
            let mut messages = Vec::new();
            for item in history {
                match item {
                    ResponseItem::Message { role, content, .. } => {
                        let r = match role {
                            Role::System => "system",
                            Role::Assistant => "assistant",
                            Role::Tool => "tool",
                            Role::User => "user",
                        };
                        let text = content.iter().filter_map(|p| match p {
                            ContentPart::Text { text } => Some(text.clone()),
                            _ => None,
                        }).collect::<Vec<_>>().join("\n");
                        messages.push(llm_api::chat::Message {
                            role: r.to_string(),
                            content: Some(text),
                            tool_call_id: None,
                            tool_calls: None,
                        });
                    }
                    ResponseItem::FunctionCall { call_id, name, arguments, .. } => {
                        messages.push(llm_api::chat::Message {
                            role: "assistant".to_string(),
                            content: None,
                            tool_call_id: None,
                            tool_calls: Some(vec![llm_api::chat::ToolCall {
                                id: call_id.clone(),
                                r#type: "function".to_string(),
                                function: llm_api::chat::FunctionCall {
                                    name: name.clone(),
                                    arguments: arguments.clone(),
                                },
                            }]),
                        });
                    }
                    ResponseItem::FunctionCallOutput { call_id, output, .. } => {
                        messages.push(llm_api::chat::Message {
                            role: "tool".to_string(),
                            content: Some(output.clone()),
                            tool_call_id: Some(call_id.clone()),
                            tool_calls: None,
                        });
                    }
                    ResponseItem::Reasoning { .. } => {}
                }
            }

            let clean_model = model_str.strip_prefix("groq/").unwrap_or(model_str);
            let llm = llm_api::chat::ChatLlmInteraction::new(endpoint, clean_model.to_string(), api_key);
            let chat_req = llm_api::chat::ChatCompletionRequest {
                model: clean_model.to_string(),
                messages,
                temperature: Some(0.7),
                max_tokens: None,
                top_p: None,
                stop: None,
                stream: None,
                tools: None,
                tool_choice: None,
            };

            let res = llm.call_chat_completions_v2(&chat_req).await
                .map_err(|e| format!("Chat completions call failed: {}", e))?;

            if let Some(choice) = res.choices.into_iter().next() {
                let mut output_items = Vec::new();
                if let Some(tool_calls) = choice.message.tool_calls {
                    for tc in tool_calls {
                        output_items.push(ResponseItem::FunctionCall {
                            id: format!("fc_{}", Uuid::new_v4()),
                            call_id: tc.id,
                            name: tc.function.name,
                            arguments: tc.function.arguments,
                        });
                    }
                }
                if let Some(content) = choice.message.content {
                    if !content.is_empty() {
                        output_items.push(ResponseItem::Message {
                            id: format!("resp_msg_{}", Uuid::new_v4()),
                            role: Role::Assistant,
                            content: vec![ContentPart::Text { text: content }],
                        });
                    }
                }
                if !output_items.is_empty() {
                    return Ok(output_items);
                }
            }
        }

        // Default mock fallback
        SimpleGatewayBackend.process_turn(session_id, history, model).await
    }
}

/// Shared Gateway State
#[derive(Clone)]
pub struct GatewayState {
    pub session_store: Arc<SessionStore>,
    pub backend: Arc<dyn GatewayBackend>,
}

pub struct GatewayServer {
    state: GatewayState,
}

impl GatewayServer {
    pub fn new(session_store: Arc<SessionStore>, backend: Arc<dyn GatewayBackend>) -> Self {
        Self {
            state: GatewayState {
                session_store,
                backend,
            },
        }
    }

    pub fn with_default_backend(session_store: Arc<SessionStore>) -> Self {
        Self::new(session_store, Arc::new(SimpleGatewayBackend))
    }

    /// Build the Axum router for the gateway
    pub fn router(&self) -> Router {
        Router::new()
            .route("/v1/responses", post(handle_responses))
            .route("/v1/chat/completions", post(handle_chat_completions))
            .with_state(self.state.clone())
    }

    /// Start the HTTP server on the given address (e.g. "0.0.0.0:8080")
    pub async fn start(&self, bind_address: &str) -> Result<(), Box<dyn std::error::Error>> {
        let listener = tokio::net::TcpListener::bind(bind_address).await?;
        tracing::info!("🚀 Swarm Gateway Server running on {}", bind_address);
        axum::serve(listener, self.router()).await?;
        Ok(())
    }
}

// -------------------------------------------------------------------------------------------------
// Route 1: POST /v1/responses (Open Responses Protocol)
// -------------------------------------------------------------------------------------------------

async fn handle_responses(
    State(state): State<GatewayState>,
    Json(payload): Json<CreateResponseRequest>,
) -> Response {
    let is_stream = payload.stream.unwrap_or(false);
    let session = state
        .session_store
        .resolve_session(payload.previous_response_id.as_deref())
        .await;

    // 1. Normalize input into ResponseItem(s)
    let input_items: Vec<ResponseItem> = match payload.input {
        Some(ResponsesInput::Text(text)) => vec![ResponseItem::Message {
            id: format!("msg_{}", Uuid::new_v4()),
            role: Role::User,
            content: vec![ContentPart::Text { text }],
        }],
        Some(ResponsesInput::Items(items)) => items,
        None => vec![],
    };

    // 2. Append input items to session
    if !input_items.is_empty() {
        state
            .session_store
            .append_items(&session.id, &input_items)
            .await;
    }

    // 3. Retrieve full history
    let history = state.session_store.get_history(&session.id).await;

    // 4. Process through backend
    let model_name = payload.model.clone().unwrap_or_else(|| "default-swarm-model".to_string());
    let output_items = match state
        .backend
        .process_turn(&session.id, &history, payload.model.as_deref())
        .await
    {
        Ok(items) => items,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": err })),
            )
                .into_response();
        }
    };

    // 5. Append output items to session and track parent response ID
    if let Some(last_item) = output_items.last() {
        let last_id = match last_item {
            ResponseItem::Message { id, .. } => id.clone(),
            ResponseItem::Reasoning { id, .. } => id.clone(),
            ResponseItem::FunctionCall { id, .. } => id.clone(),
            ResponseItem::FunctionCallOutput { id, .. } => id.clone(),
        };
        state
            .session_store
            .set_parent_response_id(&session.id, last_id)
            .await;
    }
    state
        .session_store
        .append_items(&session.id, &output_items)
        .await;

    let response_id = format!("resp_{}", Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if is_stream {
        // SSE streaming response emitting delta ResponseItem events
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();
        for item in output_items {
            let json_str = serde_json::to_string(&item).unwrap_or_default();
            events.push(Ok(Event::default().event("response.item").data(json_str)));
        }
        events.push(Ok(Event::default().data("[DONE]")));

        let stream = stream::iter(events);
        Sse::new(stream).into_response()
    } else {
        let response_obj = ResponseObject {
            id: response_id,
            object: "response".to_string(),
            created,
            model: model_name,
            output: output_items,
            usage: Some(ResponseUsage {
                input_tokens: 10,
                output_tokens: 20,
                total_tokens: 30,
            }),
        };
        Json(response_obj).into_response()
    }
}

// -------------------------------------------------------------------------------------------------
// Route 2: POST /v1/chat/completions (Stateless Chat Completions Normalization)
// -------------------------------------------------------------------------------------------------

async fn handle_chat_completions(
    State(state): State<GatewayState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let session_id = format!("stateless_chat_{}", Uuid::new_v4());
    let is_stream = payload.stream.unwrap_or(false);

    // 1. Normalize OpenAI messages into internal ResponseItems
    let mut normalized_items = Vec::new();
    for msg in payload.messages {
        let role = match msg.role.to_lowercase().as_str() {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        };

        if let Some(tool_calls) = msg.tool_calls {
            for tc in tool_calls {
                normalized_items.push(ResponseItem::FunctionCall {
                    id: format!("fc_{}", Uuid::new_v4()),
                    call_id: tc.id,
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                });
            }
        } else if role == Role::Tool {
            normalized_items.push(ResponseItem::FunctionCallOutput {
                id: format!("fco_{}", Uuid::new_v4()),
                call_id: msg.tool_call_id.unwrap_or_else(|| "call_default".to_string()),
                output: msg.content.unwrap_or_default(),
                is_error: false,
            });
        } else if let Some(content) = msg.content {
            normalized_items.push(ResponseItem::Message {
                id: format!("msg_{}", Uuid::new_v4()),
                role,
                content: vec![ContentPart::Text { text: content }],
            });
        }
    }

    // 2. Append to session store
    state
        .session_store
        .append_items(&session_id, &normalized_items)
        .await;

    // 3. Process with backend
    let output_items = match state
        .backend
        .process_turn(&session_id, &normalized_items, Some(&payload.model))
        .await
    {
        Ok(items) => items,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": err })),
            )
                .into_response();
        }
    };

    // 4. Normalize ResponseItems back into standard ChatCompletionResponse
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut response_text = String::new();
    let mut tool_calls = Vec::new();

    for item in &output_items {
        match item {
            ResponseItem::Message { content, .. } => {
                for part in content {
                    if let ContentPart::Text { text } = part {
                        response_text.push_str(text);
                    }
                }
            }
            ResponseItem::FunctionCall { call_id, name, arguments, .. } => {
                tool_calls.push(llm_api::chat::ToolCall {
                    id: call_id.clone(),
                    r#type: "function".to_string(),
                    function: llm_api::chat::FunctionCall {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                });
            }
            _ => {}
        }
    }

    let finish_reason = if !tool_calls.is_empty() {
        "tool_calls".to_string()
    } else {
        "stop".to_string()
    };

    let tool_calls_opt = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    let content_opt = if response_text.is_empty() && tool_calls_opt.is_some() {
        None
    } else {
        Some(response_text.clone())
    };

    if is_stream {
        // SSE streaming chunk for ChatCompletion
        #[derive(Serialize)]
        struct ChatChunk {
            id: String,
            object: String,
            created: u64,
            model: String,
            choices: Vec<ChatChunkChoice>,
        }
        #[derive(Serialize)]
        struct ChatChunkChoice {
            index: u32,
            delta: ChatChunkDelta,
            finish_reason: Option<String>,
        }
        #[derive(Serialize)]
        struct ChatChunkDelta {
            #[serde(skip_serializing_if = "Option::is_none")]
            role: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            content: Option<String>,
        }

        let chunk1 = ChatChunk {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion.chunk".to_string(),
            created,
            model: payload.model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatChunkDelta {
                    role: Some("assistant".to_string()),
                    content: Some(response_text),
                },
                finish_reason: None,
            }],
        };

        let chunk_final = ChatChunk {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion.chunk".to_string(),
            created,
            model: payload.model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatChunkDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(finish_reason),
            }],
        };

        let events: Vec<Result<Event, Infallible>> = vec![
            Ok(Event::default().data(serde_json::to_string(&chunk1).unwrap_or_default())),
            Ok(Event::default().data(serde_json::to_string(&chunk_final).unwrap_or_default())),
            Ok(Event::default().data("[DONE]")),
        ];

        let stream = stream::iter(events);
        Sse::new(stream).into_response()
    } else {
        let chat_response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created,
            model: payload.model,
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: content_opt,
                    tool_calls: tool_calls_opt,
                },
                logprobs: None,
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens: 15,
                completion_tokens: 25,
                total_tokens: 40,
            },
            system_fingerprint: None,
        };

        Json(chat_response).into_response()
    }
}
