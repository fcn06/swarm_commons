use std::sync::Arc;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt; // for oneshot

use agent_core::server::gateway_server::GatewayServer;
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
