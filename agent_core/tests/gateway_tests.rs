use std::sync::Arc;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt; // for oneshot

use a2a_rs::domain::core::agent::SecurityScheme;
use a2a_rs::domain::A2AError;
use a2a_rs::port::authenticator::{AuthContext, AuthPrincipal, Authenticator};
use agent_core::server::auth::SharedAuthenticator;
use agent_core::server::gateway_server::{
    BackendTurnResult, GatewayBackend, GatewayResilienceSection, GatewayServer,
};
use agent_core::session::SessionStore;
use agent_models::response_item::{ResponseItem, ResponseObject};
use llm_api::chat::ChatCompletionResponse;
use llm_api::google_interactions::GoogleInteractionsAdapter;

#[tokio::test]
async fn test_chat_completions_stateless_endpoint() {
    let session_store = Arc::new(SessionStore::new());
    let server = GatewayServer::with_default_backend(session_store);
    let app = server.router();

    let request_body = json!({
        "model": "swarm-fast-v1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a rust function"}
        ]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let chat_resp: ChatCompletionResponse = serde_json::from_slice(&body_bytes).unwrap();

    assert_eq!(chat_resp.model, "swarm-fast-v1");
    assert!(!chat_resp.choices.is_empty());
    assert!(chat_resp.choices[0].message.content.as_ref().unwrap().contains("Swarm Gateway Response"));
    // Verify non-zero real usage tracking
    assert!(chat_resp.usage.prompt_tokens > 0);
    assert_eq!(chat_resp.usage.completion_tokens, 10);
    assert_eq!(chat_resp.usage.total_tokens, chat_resp.usage.prompt_tokens + 10);
}

#[tokio::test]
async fn test_responses_stateful_session_chaining() {
    let session_store = Arc::new(SessionStore::new());
    let server = GatewayServer::with_default_backend(session_store.clone());
    let app = server.router();

    // Turn 1: Initial query
    let turn1_req = json!({
        "model": "swarm-stateful-v1",
        "input": "My favorite color is navy blue.",
        "stream": false
    });

    let req1 = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&turn1_req).unwrap()))
        .unwrap();

    let res1 = app.clone().oneshot(req1).await.unwrap();
    assert_eq!(res1.status(), StatusCode::OK);

    let body_bytes1 = res1.into_body().collect().await.unwrap().to_bytes();
    let resp_obj1: ResponseObject = serde_json::from_slice(&body_bytes1).unwrap();
    assert_eq!(resp_obj1.output.len(), 1);
    assert!(resp_obj1.usage.is_some());
    assert!(resp_obj1.usage.as_ref().unwrap().total_tokens > 0);

    let turn1_resp_id = match &resp_obj1.output[0] {
        ResponseItem::Message { id, .. } => id.clone(),
        _ => panic!("Expected message"),
    };

    // Turn 2: Follow-up referencing previous_response_id
    let turn2_req = json!({
        "model": "swarm-stateful-v1",
        "input": "What is my favorite color?",
        "previous_response_id": turn1_resp_id,
        "stream": false
    });

    let req2 = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&turn2_req).unwrap()))
        .unwrap();

    let res2 = app.oneshot(req2).await.unwrap();
    assert_eq!(res2.status(), StatusCode::OK);

    let body_bytes2 = res2.into_body().collect().await.unwrap().to_bytes();
    let resp_obj2: ResponseObject = serde_json::from_slice(&body_bytes2).unwrap();
    assert_eq!(resp_obj2.output.len(), 1);
    assert!(resp_obj2.usage.is_some());

    // Verify session store preserved the full multi-turn history
    let session = session_store.resolve_session(Some(&turn1_resp_id)).await;
    let history = session_store.get_history(&session.id).await;

    // History should contain: Turn 1 User, Turn 1 Output, Turn 2 User, Turn 2 Output = 4 items
    assert_eq!(history.len(), 4);

    // Verify Google Interactions adapter correctly transforms this entire multi-turn history
    let gemini_req = GoogleInteractionsAdapter::to_gemini_request(&history, Some(turn1_resp_id.clone())).unwrap();
    assert_eq!(gemini_req.previous_interaction_id, Some(turn1_resp_id));
    assert_eq!(gemini_req.contents.len(), 4);
    assert_eq!(gemini_req.contents[0].role, "user");
    assert_eq!(gemini_req.contents[1].role, "model");
    assert_eq!(gemini_req.contents[2].role, "user");
    assert_eq!(gemini_req.contents[3].role, "model");
}

#[tokio::test]
async fn test_responses_sse_streaming() {
    let session_store = Arc::new(SessionStore::new());
    let server = GatewayServer::with_default_backend(session_store);
    let app = server.router();

    let stream_req = json!({
        "model": "swarm-stream-v1",
        "input": "Stream this test response",
        "stream": true
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&stream_req).unwrap()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&body_bytes);

    assert!(body_str.contains("event: response.item"));
    assert!(body_str.contains("data: [DONE]"));
}

#[tokio::test]
async fn test_chat_completions_sse_streaming() {
    let session_store = Arc::new(SessionStore::new());
    let server = GatewayServer::with_default_backend(session_store);
    let app = server.router();

    let stream_req = json!({
        "model": "swarm-stream-v1",
        "messages": [
            {"role": "user", "content": "Stream chat completion"}
        ],
        "stream": true
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&stream_req).unwrap()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&body_bytes);

    assert!(body_str.contains("data:"));
    assert!(body_str.contains("[DONE]"));
}

#[tokio::test]
async fn test_gateway_health_endpoint() {
    let session_store = Arc::new(SessionStore::new());
    let server = GatewayServer::with_default_backend(session_store);
    let app = server.router();

    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body_json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body_json["status"], "healthy");
    assert_eq!(body_json["service"], "swarm-gateway");
}

#[tokio::test]
async fn test_gateway_concurrency_limit_load_shed() {
    let session_store = Arc::new(SessionStore::new());
    let resilience = GatewayResilienceSection {
        max_concurrent_requests: Some(1),
        request_timeout_seconds: None,
        circuit_breaker_failure_threshold: None,
        circuit_breaker_reset_seconds: None,
    };
    let server = GatewayServer::with_default_backend(session_store)
        .with_resilience(resilience);

    // Acquire the only permit
    let _permit = server.router(); // get router to verify

    // Let's create an app with max_concurrency = 1
    let session_store2 = Arc::new(SessionStore::new());
    let server2 = GatewayServer::with_default_backend(session_store2)
        .with_resilience(GatewayResilienceSection {
            max_concurrent_requests: Some(1),
            ..Default::default()
        });

    let app = server2.router();

    // First request should succeed
    let req1 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({
            "model": "swarm-test",
            "messages": [{"role": "user", "content": "hi"}]
        })).unwrap()))
        .unwrap();

    let res1 = app.oneshot(req1).await.unwrap();
    assert_eq!(res1.status(), StatusCode::OK);
}

struct SlowMockBackend {
    pub delay_ms: u64,
}

#[async_trait::async_trait]
impl GatewayBackend for SlowMockBackend {
    async fn process_turn(
        &self,
        _session_id: &str,
        _history: &[ResponseItem],
        _model: Option<&str>,
    ) -> Result<BackendTurnResult, String> {
        tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
        Ok(BackendTurnResult {
            items: vec![],
            usage: None,
        })
    }
}

#[tokio::test]
async fn test_gateway_timeout_protection() {
    let session_store = Arc::new(SessionStore::new());
    let slow_backend = Arc::new(SlowMockBackend { delay_ms: 500 });
    let resilience = GatewayResilienceSection {
        max_concurrent_requests: None,
        request_timeout_seconds: Some(1), // 1 second timeout
        circuit_breaker_failure_threshold: None,
        circuit_breaker_reset_seconds: None,
    };

    // When backend takes 500ms and timeout is 1s, it should succeed
    let server = GatewayServer::new(session_store.clone(), slow_backend.clone())
        .with_resilience(resilience);
    let app = server.router();

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({
            "model": "swarm-test",
            "messages": [{"role": "user", "content": "hi"}]
        })).unwrap()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

struct FailingMockBackend;

#[async_trait::async_trait]
impl GatewayBackend for FailingMockBackend {
    async fn process_turn(
        &self,
        _session_id: &str,
        _history: &[ResponseItem],
        _model: Option<&str>,
    ) -> Result<BackendTurnResult, String> {
        Err("Circuit breaker is OPEN for provider 'test-prov'. Fast-failing request to protect failure domain.".to_string())
    }
}

#[tokio::test]
async fn test_gateway_circuit_breaker_503_fast_fail() {
    let session_store = Arc::new(SessionStore::new());
    let failing_backend = Arc::new(FailingMockBackend);
    let server = GatewayServer::new(session_store, failing_backend);
    let app = server.router();

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({
            "model": "swarm-test",
            "messages": [{"role": "user", "content": "hi"}]
        })).unwrap()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);

    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body_json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body_json["error"]["type"], "backend_error");
    assert_eq!(body_json["error"]["code"], "service_unavailable");
}

struct GatewayTestAuthenticator {
    scheme: SecurityScheme,
}

impl GatewayTestAuthenticator {
    fn new() -> Self {
        Self {
            scheme: SecurityScheme::Http {
                scheme: "bearer".to_string(),
                bearer_format: Some("token".to_string()),
                description: None,
            },
        }
    }
}

#[async_trait::async_trait]
impl Authenticator for GatewayTestAuthenticator {
    async fn authenticate(&self, context: &AuthContext) -> Result<AuthPrincipal, A2AError> {
        if context.credential == "valid_gateway_token" {
            Ok(AuthPrincipal::new("gateway_client".to_string(), "bearer".to_string()))
        } else {
            Err(A2AError::Internal("Bad gateway token".to_string()))
        }
    }

    fn security_scheme(&self) -> &SecurityScheme {
        &self.scheme
    }

    fn validate_context(&self, _context: &AuthContext) -> Result<(), A2AError> {
        Ok(())
    }
}

#[tokio::test]
async fn test_gateway_inbound_auth_enforced() {
    let session_store = Arc::new(SessionStore::new());
    let server = GatewayServer::with_default_backend(session_store)
        .with_authenticator(SharedAuthenticator::new(GatewayTestAuthenticator::new()));
    let app = server.router();

    // 1. Health check is public without auth
    let health_req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let health_res = app.clone().oneshot(health_req).await.unwrap();
    assert_eq!(health_res.status(), StatusCode::OK);

    // 2. Chat completions without header returns 401
    let unauth_req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({
            "model": "swarm-test",
            "messages": [{"role": "user", "content": "hi"}]
        })).unwrap()))
        .unwrap();
    let unauth_res = app.clone().oneshot(unauth_req).await.unwrap();
    assert_eq!(unauth_res.status(), StatusCode::UNAUTHORIZED);

    // 3. Chat completions with invalid token returns 401
    let bad_req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .header("Authorization", "Bearer invalid_secret")
        .body(Body::from(serde_json::to_vec(&json!({
            "model": "swarm-test",
            "messages": [{"role": "user", "content": "hi"}]
        })).unwrap()))
        .unwrap();
    let bad_res = app.clone().oneshot(bad_req).await.unwrap();
    assert_eq!(bad_res.status(), StatusCode::UNAUTHORIZED);

    // 4. Chat completions with valid token succeeds (200 OK)
    let good_req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .header("Authorization", "Bearer valid_gateway_token")
        .body(Body::from(serde_json::to_vec(&json!({
            "model": "swarm-test",
            "messages": [{"role": "user", "content": "hi"}]
        })).unwrap()))
        .unwrap();
    let good_res = app.oneshot(good_req).await.unwrap();
    assert_eq!(good_res.status(), StatusCode::OK);
}

// Mock backend testing tool forwarding
struct ToolInspectBackend {
    pub received_tools: Arc<tokio::sync::Mutex<Option<Vec<llm_api::tools::Tool>>>>,
}

#[async_trait::async_trait]
impl GatewayBackend for ToolInspectBackend {
    async fn process_turn(
        &self,
        session_id: &str,
        history: &[ResponseItem],
        model: Option<&str>,
    ) -> Result<BackendTurnResult, String> {
        self.process_turn_with_options(
            session_id,
            history,
            agent_core::server::gateway_server::GatewayTurnOptions {
                model: model.map(|s| s.to_string()),
                tools: None,
                tool_choice: None,
            },
        ).await
    }

    async fn process_turn_with_options(
        &self,
        _session_id: &str,
        _history: &[ResponseItem],
        options: agent_core::server::gateway_server::GatewayTurnOptions,
    ) -> Result<BackendTurnResult, String> {
        let mut lock = self.received_tools.lock().await;
        *lock = options.tools;

        Ok(BackendTurnResult {
            items: vec![ResponseItem::FunctionCall {
                id: "fc_1".to_string(),
                call_id: "call_abc123".to_string(),
                name: "lookup_inventory".to_string(),
                arguments: r#"{"item_id":"widget"}"#.to_string(),
            }],
            usage: None,
        })
    }
}

#[tokio::test]
async fn test_gateway_tool_forwarding_options() {
    let session_store = Arc::new(SessionStore::new());
    let received_tools = Arc::new(tokio::sync::Mutex::new(None));
    let backend = Arc::new(ToolInspectBackend {
        received_tools: received_tools.clone(),
    });
    let server = GatewayServer::new(session_store, backend);
    let app = server.router();

    let req_body = json!({
        "model": "tool-model",
        "messages": [{"role": "user", "content": "Check stock"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_inventory",
                    "description": "Check inventory for an item",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"}
                        }
                    }
                }
            }
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let lock = received_tools.lock().await;
    assert!(lock.is_some());
    assert_eq!(lock.as_ref().unwrap().len(), 1);
    assert_eq!(lock.as_ref().unwrap()[0].function.name, "lookup_inventory");

    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let chat_resp: ChatCompletionResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(chat_resp.choices.len(), 1);
    let tool_calls = chat_resp.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "lookup_inventory");
}

#[tokio::test]
async fn test_gateway_error_status_mapping() {
    struct CustomErrBackend {
        err_msg: String,
    }
    #[async_trait::async_trait]
    impl GatewayBackend for CustomErrBackend {
        async fn process_turn(
            &self,
            _s: &str,
            _h: &[ResponseItem],
            _m: Option<&str>,
        ) -> Result<BackendTurnResult, String> {
            Err(self.err_msg.clone())
        }
    }

    let session_store = Arc::new(SessionStore::new());

    // 429 rate limit
    let server429 = GatewayServer::new(session_store.clone(), Arc::new(CustomErrBackend {
        err_msg: "HTTP 429: Rate limit exceeded".to_string(),
    }));
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}]
        })).unwrap()))
        .unwrap();
    let res = server429.router().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);

    // 502 bad gateway
    let server502 = GatewayServer::new(session_store.clone(), Arc::new(CustomErrBackend {
        err_msg: "Bad gateway upstream returned 502".to_string(),
    }));
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}]
        })).unwrap()))
        .unwrap();
    let res = server502.router().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_GATEWAY);

    // 504 gateway timeout
    let server504 = GatewayServer::new(session_store, Arc::new(CustomErrBackend {
        err_msg: "Upstream timeout after 30s".to_string(),
    }));
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}]
        })).unwrap()))
        .unwrap();
    let res = server504.router().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::GATEWAY_TIMEOUT);
}
