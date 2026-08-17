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
use serde::Serialize;
use uuid::Uuid;

use agent_models::response_item::{
    ContentPart, CreateResponseRequest, ResponseItem, ResponseObject, ResponseUsage, ResponsesInput, Role,
};
use llm_api::chat::{
    ChatCompletionRequest, ChatCompletionResponse, Choice, ResponseMessage, Usage,
};

use crate::session::SessionStoreApi;

/// Usage data returned from a backend turn
#[derive(Debug, Clone, Default, Serialize, serde::Deserialize, PartialEq)]
pub struct BackendUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

/// Result of a single backend processing turn
#[derive(Debug, Clone)]
pub struct BackendTurnResult {
    pub items: Vec<ResponseItem>,
    pub usage: Option<BackendUsage>,
}

/// Trait for handling the gateway generation backend (e.g. LLM call, agent orchestration loop)
#[async_trait::async_trait]
pub trait GatewayBackend: Send + Sync {
    async fn process_turn(
        &self,
        session_id: &str,
        history: &[ResponseItem],
        model: Option<&str>,
    ) -> Result<BackendTurnResult, String>;

    /// Stream response chunks. Default implementation calls process_turn and sends items as chunks.
    async fn process_turn_stream(
        &self,
        session_id: &str,
        history: &[ResponseItem],
        model: Option<&str>,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<Option<BackendUsage>, String> {
        let result = self.process_turn(session_id, history, model).await?;
        for item in &result.items {
            let json_str = serde_json::to_string(item).unwrap_or_default();
            let _ = tx.send(json_str).await;
        }
        let _ = tx.send("[DONE]".to_string()).await;
        Ok(result.usage)
    }
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
    ) -> Result<BackendTurnResult, String> {
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

        Ok(BackendTurnResult {
            items: vec![response_item],
            usage: Some(BackendUsage {
                input_tokens: last_user_text.split_whitespace().count() as u32,
                output_tokens: 10,
                total_tokens: last_user_text.split_whitespace().count() as u32 + 10,
            }),
        })
    }
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct GatewayConfigFile {
    pub server: Option<GatewayServerSection>,
    pub session: Option<GatewaySessionSection>,
    pub models: Option<GatewayModelsSection>,
    pub providers: Option<GatewayProvidersSection>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct GatewayServerSection {
    pub bind_address: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub log_level: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct GatewaySessionSection {
    pub max_history_items: Option<usize>,
    pub session_timeout_seconds: Option<u64>,
    pub persistence_enabled: Option<bool>,
    pub db_path: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct GatewayModelsSection {
    pub default_model: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct GatewayProvidersSection {
    pub groq: Option<GatewayProviderEntry>,
    pub google: Option<GatewayProviderEntry>,
    pub openai: Option<GatewayProviderEntry>,
    pub custom: Option<GatewayProviderEntry>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct GatewayProviderEntry {
    pub api_url: Option<String>,
    pub api_key: Option<String>,
    pub default_model: Option<String>,
    pub recommended_models: Option<Vec<String>>,
}

/// Multi-model gateway backend capable of routing to Google Gemini, Groq, OpenAI, or local models
pub struct MultiModelGatewayBackend {
    pub client: reqwest::Client,
    pub gemini_api_key: Option<String>,
    pub groq_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub custom_endpoint: Option<String>,
    pub groq_url: String,
    pub gemini_url: String,
    pub openai_url: String,
    pub default_model: Option<String>,
    pub groq_models: Vec<String>,
    pub google_models: Vec<String>,
    pub openai_models: Vec<String>,
    pub custom_models: Vec<String>,
}

fn get_env_var(key: &str) -> Option<String> {
    if let Ok(v) = std::env::var(key) {
        let trimmed = v.trim();
        if !trimmed.is_empty() && !trimmed.starts_with("your_") && !trimmed.starts_with('<') {
            return Some(trimmed.to_string());
        }
    }
    let candidate_paths = [
        ".env",
        "../.env",
        "../../.env",
        "swarm/.env",
    ];
    for path in &candidate_paths {
        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                if let Some((k, v)) = trimmed.split_once('=') {
                    if k.trim() == key {
                        let val = v.trim().trim_matches('"').trim_matches('\'');
                        if !val.is_empty() && !val.starts_with("your_") && !val.starts_with('<') {
                            return Some(val.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

impl Default for MultiModelGatewayBackend {
    fn default() -> Self {
        Self::from_env()
    }
}

impl MultiModelGatewayBackend {
    pub fn from_env() -> Self {
        let gemini_api_key = get_env_var("GEMINI_API_KEY")
            .or_else(|| get_env_var("LLM_GEMINI_API_KEY"));
        let groq_api_key = get_env_var("GROQ_API_KEY")
            .or_else(|| get_env_var("LLM_GROQ_API_KEY"))
            .or_else(|| get_env_var("LLM_API_KEY"));
        let openai_api_key = get_env_var("OPENAI_API_KEY");
        let custom_endpoint = get_env_var("SWARM_LLM_URL");

        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(45))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            gemini_api_key,
            groq_api_key,
            openai_api_key,
            custom_endpoint,
            groq_url: "https://api.groq.com/openai/v1/chat/completions".to_string(),
            gemini_url: "https://generativelanguage.googleapis.com/v1beta/models".to_string(),
            openai_url: "https://api.openai.com/v1/chat/completions".to_string(),
            default_model: None,
            groq_models: vec![
                "groq/llama-3.3-70b-versatile".to_string(),
                "openai/gpt-oss-20b".to_string(),
                "qwen/qwen3-32b".to_string(),
                "llama-3.3-70b-versatile".to_string(),
                "llama-3.1-8b-instant".to_string(),
            ],
            google_models: vec![
                "gemini-2.0-flash".to_string(),
                "gemini-1.5-pro".to_string(),
                "gemini-1.5-flash".to_string(),
                "google/gemini-2.0-flash".to_string(),
            ],
            openai_models: vec![
                "gpt-4o".to_string(),
                "gpt-4o-mini".to_string(),
                "gpt-4-turbo".to_string(),
                "gpt-3.5-turbo".to_string(),
            ],
            custom_models: vec![
                "llama3.2:latest".to_string(),
                "mistral:latest".to_string(),
                "deepseek-r1:8b".to_string(),
                "qwen2.5:latest".to_string(),
            ],
        }
    }

    pub fn from_config(config: &GatewayConfigFile) -> Self {
        let mut backend = Self::from_env();

        if let Some(models) = &config.models {
            if let Some(dm) = &models.default_model {
                backend.default_model = Some(dm.clone());
            }
        }

        if let Some(providers) = &config.providers {
            if let Some(groq) = &providers.groq {
                if let Some(url) = &groq.api_url {
                    backend.groq_url = url.clone();
                }
                if let Some(key) = &groq.api_key {
                    if !key.is_empty() && !key.starts_with('<') {
                        backend.groq_api_key = Some(key.clone());
                    }
                }
                if let Some(models) = &groq.recommended_models {
                    backend.groq_models = models.clone();
                }
            }
            if let Some(google) = &providers.google {
                if let Some(url) = &google.api_url {
                    backend.gemini_url = url.clone();
                }
                if let Some(key) = &google.api_key {
                    if !key.is_empty() && !key.starts_with('<') {
                        backend.gemini_api_key = Some(key.clone());
                    }
                }
                if let Some(models) = &google.recommended_models {
                    backend.google_models = models.clone();
                }
            }
            if let Some(openai) = &providers.openai {
                if let Some(url) = &openai.api_url {
                    backend.openai_url = url.clone();
                }
                if let Some(key) = &openai.api_key {
                    if !key.is_empty() && !key.starts_with('<') {
                        backend.openai_api_key = Some(key.clone());
                    }
                }
                if let Some(models) = &openai.recommended_models {
                    backend.openai_models = models.clone();
                }
            }
            if let Some(custom) = &providers.custom {
                if let Some(url) = &custom.api_url {
                    backend.custom_endpoint = Some(url.clone());
                }
                if let Some(models) = &custom.recommended_models {
                    backend.custom_models = models.clone();
                }
            }
        }

        backend
    }

    pub fn from_config_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: GatewayConfigFile = toml::from_str(&content)?;
        Ok(Self::from_config(&config))
    }
}

#[async_trait::async_trait]
impl GatewayBackend for MultiModelGatewayBackend {
    async fn process_turn(
        &self,
        session_id: &str,
        history: &[ResponseItem],
        model: Option<&str>,
    ) -> Result<BackendTurnResult, String> {
        let model_str = model
            .or_else(|| self.default_model.as_deref())
            .unwrap_or("groq/llama-3.3-70b-versatile");

        // 1. Google Gemini routing via GoogleInteractionsAdapter
        let is_gemini = self.google_models.iter().any(|m| m.eq_ignore_ascii_case(model_str))
            || model_str.contains("gemini")
            || model_str.starts_with("google/");

        if is_gemini {
            if let Some(key) = &self.gemini_api_key {
                let gemini_req = llm_api::google_interactions::GoogleInteractionsAdapter::to_gemini_request(
                    history,
                    Some(session_id.to_string()),
                ).map_err(|e| format!("Failed to build Gemini interaction request: {}", e))?;

                let target_model = model_str.strip_prefix("google/").unwrap_or(model_str);
                let base_url = self.gemini_url.trim_end_matches('/');
                let url = format!("{}/{}:generateContent?key={}", base_url, target_model, key);

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
                #[serde(rename_all = "camelCase")]
                struct GeminiUsageMetadata {
                    prompt_token_count: Option<u32>,
                    candidates_token_count: Option<u32>,
                    total_token_count: Option<u32>,
                }

                #[derive(serde::Deserialize)]
                #[serde(rename_all = "camelCase")]
                struct GeminiResponse {
                    candidates: Option<Vec<GeminiCandidate>>,
                    usage_metadata: Option<GeminiUsageMetadata>,
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
                    let usage = gemini_data.usage_metadata.map(|u| BackendUsage {
                        input_tokens: u.prompt_token_count.unwrap_or(0),
                        output_tokens: u.candidates_token_count.unwrap_or(0),
                        total_tokens: u.total_token_count.unwrap_or(0),
                    });
                    return Ok(BackendTurnResult {
                        items: output_items,
                        usage,
                    });
                }
            } else {
                return Err(format!("Gemini API key not found. Please set GEMINI_API_KEY to use model '{}'.", model_str));
            }
        }

        // 2. OpenAI / Groq / Custom chat completion routing
        let is_groq = self.groq_models.iter().any(|m| m.eq_ignore_ascii_case(model_str))
            || model_str.starts_with("groq/")
            || model_str == "openai/gpt-oss-20b"
            || model_str.starts_with("qwen/");

        let is_openai = self.openai_models.iter().any(|m| m.eq_ignore_ascii_case(model_str))
            || model_str.starts_with("gpt-")
            || model_str.starts_with("o1")
            || model_str.starts_with("o3");

        let is_custom = self.custom_models.iter().any(|m| m.eq_ignore_ascii_case(model_str))
            || model_str.contains(':')
            || model_str.starts_with("ollama/")
            || model_str.starts_with("local/");

        let (endpoint, api_key, target_model) = if is_groq && self.groq_api_key.is_some() {
            (
                self.groq_url.clone(),
                self.groq_api_key.clone().unwrap(),
                model_str.strip_prefix("groq/").unwrap_or(model_str).to_string(),
            )
        } else if is_openai && self.openai_api_key.is_some() {
            (
                self.openai_url.clone(),
                self.openai_api_key.clone().unwrap(),
                model_str.strip_prefix("openai/").unwrap_or(model_str).to_string(),
            )
        } else if is_custom {
            (
                self.custom_endpoint.clone().unwrap_or_else(|| "http://localhost:11434/v1/chat/completions".to_string()),
                self.openai_api_key.clone().or_else(|| self.groq_api_key.clone()).unwrap_or_default(),
                model_str.strip_prefix("ollama/").or_else(|| model_str.strip_prefix("local/")).unwrap_or(model_str).to_string(),
            )
        } else if let Some(groq_key) = &self.groq_api_key {
            // Default to Groq when GROQ_API_KEY is available
            (
                self.groq_url.clone(),
                groq_key.clone(),
                model_str.strip_prefix("groq/").unwrap_or(model_str).to_string(),
            )
        } else if let Some(openai_key) = &self.openai_api_key {
            // Default to OpenAI when OPENAI_API_KEY is available
            (
                self.openai_url.clone(),
                openai_key.clone(),
                model_str.strip_prefix("openai/").unwrap_or(model_str).to_string(),
            )
        } else if let Some(custom_url) = &self.custom_endpoint {
            // Fall back to custom / local endpoint
            (
                custom_url.clone(),
                String::new(),
                model_str.strip_prefix("ollama/").or_else(|| model_str.strip_prefix("local/")).unwrap_or(model_str).to_string(),
            )
        } else {
            return Err(format!("No configured provider or API key found to handle model '{}'. Set GROQ_API_KEY, OPENAI_API_KEY, or SWARM_LLM_URL.", model_str));
        };

        if !endpoint.is_empty() {
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

            let llm = llm_api::chat::ChatLlmInteraction::new(endpoint, target_model.clone(), api_key);
            let chat_req = llm_api::chat::ChatCompletionRequest {
                model: target_model,
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

            let usage = Some(BackendUsage {
                input_tokens: res.usage.prompt_tokens,
                output_tokens: res.usage.completion_tokens,
                total_tokens: res.usage.total_tokens,
            });

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
                    return Ok(BackendTurnResult {
                        items: output_items,
                        usage,
                    });
                }
            }
        }

        // Default mock fallback
        SimpleGatewayBackend.process_turn(session_id, history, model).await
    }

    async fn process_turn_stream(
        &self,
        session_id: &str,
        history: &[ResponseItem],
        model: Option<&str>,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<Option<BackendUsage>, String> {
        let model_str = model
            .or_else(|| self.default_model.as_deref())
            .unwrap_or("groq/llama-3.3-70b-versatile");

        let is_gemini = self.google_models.iter().any(|m| m.eq_ignore_ascii_case(model_str))
            || model_str.contains("gemini")
            || model_str.starts_with("google/");

        if is_gemini {
            // Fallback for Gemini to generate items and stream as response.item
            let result = self.process_turn(session_id, history, model).await?;
            for item in &result.items {
                let json_str = serde_json::to_string(item).unwrap_or_default();
                let _ = tx.send(json_str).await;
            }
            let _ = tx.send("[DONE]".to_string()).await;
            return Ok(result.usage);
        }

        let is_groq = self.groq_models.iter().any(|m| m.eq_ignore_ascii_case(model_str))
            || model_str.starts_with("groq/")
            || model_str == "openai/gpt-oss-20b"
            || model_str.starts_with("qwen/");

        let is_openai = self.openai_models.iter().any(|m| m.eq_ignore_ascii_case(model_str))
            || model_str.starts_with("gpt-")
            || model_str.starts_with("o1")
            || model_str.starts_with("o3");

        let is_custom = self.custom_models.iter().any(|m| m.eq_ignore_ascii_case(model_str))
            || model_str.contains(':')
            || model_str.starts_with("ollama/")
            || model_str.starts_with("local/");

        let (endpoint, api_key, target_model) = if is_groq && self.groq_api_key.is_some() {
            (
                self.groq_url.clone(),
                self.groq_api_key.clone().unwrap(),
                model_str.strip_prefix("groq/").unwrap_or(model_str).to_string(),
            )
        } else if is_openai && self.openai_api_key.is_some() {
            (
                self.openai_url.clone(),
                self.openai_api_key.clone().unwrap(),
                model_str.strip_prefix("openai/").unwrap_or(model_str).to_string(),
            )
        } else if is_custom {
            (
                self.custom_endpoint.clone().unwrap_or_else(|| "http://localhost:11434/v1/chat/completions".to_string()),
                self.openai_api_key.clone().or_else(|| self.groq_api_key.clone()).unwrap_or_default(),
                model_str.strip_prefix("ollama/").or_else(|| model_str.strip_prefix("local/")).unwrap_or(model_str).to_string(),
            )
        } else if let Some(groq_key) = &self.groq_api_key {
            (
                self.groq_url.clone(),
                groq_key.clone(),
                model_str.strip_prefix("groq/").unwrap_or(model_str).to_string(),
            )
        } else if let Some(openai_key) = &self.openai_api_key {
            (
                self.openai_url.clone(),
                openai_key.clone(),
                model_str.strip_prefix("openai/").unwrap_or(model_str).to_string(),
            )
        } else if let Some(custom_url) = &self.custom_endpoint {
            (
                custom_url.clone(),
                String::new(),
                model_str.strip_prefix("ollama/").or_else(|| model_str.strip_prefix("local/")).unwrap_or(model_str).to_string(),
            )
        } else {
            return Err(format!("No configured provider found for streaming model '{}'.", model_str));
        };

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

        let llm = llm_api::chat::ChatLlmInteraction::new(endpoint, target_model.clone(), api_key);
        let chat_req = llm_api::chat::ChatCompletionRequest {
            model: target_model,
            messages,
            temperature: Some(0.7),
            max_tokens: None,
            top_p: None,
            stop: None,
            stream: Some(true),
            tools: None,
            tool_choice: None,
        };

        let usage = llm.call_chat_completions_stream(&chat_req, tx).await
            .map_err(|e| format!("Streaming chat completions failed: {}", e))?;

        Ok(usage.map(|u| BackendUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }))
    }
}

/// Shared Gateway State
#[derive(Clone)]
pub struct GatewayState {
    pub session_store: Arc<dyn SessionStoreApi>,
    pub backend: Arc<dyn GatewayBackend>,
}

pub struct GatewayServer {
    state: GatewayState,
}

impl GatewayServer {
    pub fn new(session_store: Arc<dyn SessionStoreApi>, backend: Arc<dyn GatewayBackend>) -> Self {
        Self {
            state: GatewayState {
                session_store,
                backend,
            },
        }
    }

    pub fn with_default_backend(session_store: Arc<dyn SessionStoreApi>) -> Self {
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

    if is_stream {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
        let backend = state.backend.clone();
        let session_id_clone = session.id.clone();
        let history_clone = history.clone();
        let model_clone = payload.model.clone();

        tokio::spawn(async move {
            let _ = backend.process_turn_stream(
                &session_id_clone,
                &history_clone,
                model_clone.as_deref(),
                tx,
            ).await;
        });

        let stream = async_stream::stream! {
            while let Some(chunk) = rx.recv().await {
                if chunk == "[DONE]" {
                    yield Ok::<_, Infallible>(Event::default().data("[DONE]"));
                    break;
                }
                yield Ok::<_, Infallible>(Event::default().event("response.item").data(chunk));
            }
        };
        Sse::new(stream).into_response()
    } else {
        let turn_result = match state
            .backend
            .process_turn(&session.id, &history, payload.model.as_deref())
            .await
        {
            Ok(res) => res,
            Err(err) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": err })),
                )
                    .into_response();
            }
        };

        let output_items = turn_result.items;

        // 5. Append output items to session and track parent response ID
        let response_id = format!("resp_{}", Uuid::new_v4());
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
            .set_parent_response_id(&session.id, response_id.clone())
            .await;
        state
            .session_store
            .append_items(&session.id, &output_items)
            .await;

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let response_obj = ResponseObject {
            id: response_id,
            object: "response".to_string(),
            created,
            model: model_name,
            output: output_items,
            usage: turn_result.usage.map(|u| ResponseUsage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                total_tokens: u.total_tokens,
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

    if is_stream {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
        let backend = state.backend.clone();
        let session_id_clone = session_id.clone();
        let history_clone = normalized_items.clone();
        let model_clone = payload.model.clone();

        tokio::spawn(async move {
            let _ = backend.process_turn_stream(
                &session_id_clone,
                &history_clone,
                Some(&model_clone),
                tx,
            ).await;
        });

        let stream = async_stream::stream! {
            while let Some(chunk) = rx.recv().await {
                if chunk == "[DONE]" {
                    yield Ok::<_, Infallible>(Event::default().data("[DONE]"));
                    break;
                }
                yield Ok::<_, Infallible>(Event::default().data(chunk));
            }
        };
        Sse::new(stream).into_response()
    } else {
        // 3. Process with backend
        let turn_result = match state
            .backend
            .process_turn(&session_id, &normalized_items, Some(&payload.model))
            .await
        {
            Ok(res) => res,
            Err(err) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": err })),
                )
                    .into_response();
            }
        };

        let output_items = turn_result.items;

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
            usage: turn_result.usage.map(|u| Usage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.total_tokens,
            }).unwrap_or(Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            }),
            system_fingerprint: None,
        };

        Json(chat_response).into_response()
    }
}
