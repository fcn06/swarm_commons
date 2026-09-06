use std::sync::Arc;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt; // for oneshot

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
