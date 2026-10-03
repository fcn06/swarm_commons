#![cfg(feature = "nats")]

use std::sync::Arc;
use std::time::Duration;
use a2a_rs::domain::core::agent::SecurityScheme;
use a2a_rs::domain::A2AError;
use a2a_rs::port::authenticator::{AuthContext, AuthPrincipal, Authenticator};
use a2a_rs::services::server::AsyncA2ARequestProcessor;
use a2a_rs::application::{JSONRPCResponse, json_rpc::A2ARequest};
use agent_core::server::auth::SharedAuthenticator;
use agent_core::transport::nats::{serve_nats, DefaultCredentialExtractor, NatsDispatchConfig};
use async_trait::async_trait;

#[derive(Clone)]
struct MockProcessor;

#[async_trait]
impl AsyncA2ARequestProcessor for MockProcessor {
    async fn process_raw_request(&self, request: &str) -> Result<String, A2AError> {
        let v: serde_json::Value = serde_json::from_str(request).unwrap_or_default();
        let id = v.get("id").cloned().unwrap_or(serde_json::Value::Null);
        let resp = serde_json::json!({
            "jsonrpc": "2.0",
            "result": {
                "echo": request
            },
            "id": id
        });
        Ok(serde_json::to_string(&resp).unwrap())
    }

    async fn process_request(&self, _request: &A2ARequest) -> Result<JSONRPCResponse, A2AError> {
        Err(A2AError::Internal("Not used in raw test".to_string()))
    }
}

#[derive(Clone)]
struct MockNatsAuth {
    scheme: SecurityScheme,
}

impl MockNatsAuth {
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

#[async_trait]
impl Authenticator for MockNatsAuth {
    async fn authenticate(&self, context: &AuthContext) -> Result<AuthPrincipal, A2AError> {
        if context.credential == "valid_token" {
            Ok(AuthPrincipal::new("nats_user".to_string(), "bearer".to_string()))
        } else {
            Err(A2AError::Internal("Bad NATS token".to_string()))
        }
    }

    fn security_scheme(&self) -> &SecurityScheme {
        &self.scheme
    }

    fn validate_context(&self, _context: &AuthContext) -> Result<(), A2AError> {
        Ok(())
    }
}

// Helper to spawn a real local nats-server on an ephemeral port
struct NatsTestServer {
    child: std::process::Child,
    pub port: u16,
}

impl NatsTestServer {
    fn start() -> Self {
        let port = 14222 + (std::process::id() % 1000) as u16;
        let child = std::process::Command::new("nats-server")
            .arg("-p")
            .arg(port.to_string())
            .spawn()
            .expect("Failed to start nats-server. Ensure /usr/local/bin/nats-server is installed.");
        std::thread::sleep(Duration::from_millis(300));
        Self { child, port }
    }
}

impl Drop for NatsTestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

#[tokio::test]
async fn test_nats_transport_roundtrip_and_auth() {
    let server = NatsTestServer::start();
    let nats_url = format!("127.0.0.1:{}", server.port);

    let client = async_nats::connect(&nats_url)
        .await
        .expect("Failed to connect to test nats-server");

    let cfg = NatsDispatchConfig {
        subject: "a2a.v1.test.tasks.send".to_string(),
        queue_group: Some("test_group".to_string()),
        max_in_flight: 10,
    };

    let auth = Some(SharedAuthenticator::new(MockNatsAuth::new()));
    let extractor = Arc::new(DefaultCredentialExtractor);
    let processor = MockProcessor;

    let serve_handle = tokio::spawn({
        let client = client.clone();
        async move {
            serve_nats(client, cfg, processor, auth, extractor).await.unwrap();
        }
    });

    tokio::time::sleep(Duration::from_millis(200)).await;

    // 1. Send request with valid token
    let req_valid = serde_json::json!({
        "jsonrpc": "2.0",
        "id": "req_1",
        "method": "tasks/send",
        "params": {
            "message": {
                "metadata": {
                    "agent_jwt": "valid_token"
                }
            }
        }
    });

    let resp = client
        .request("a2a.v1.test.tasks.send".to_string(), serde_json::to_vec(&req_valid).unwrap().into())
        .await
        .expect("Request should receive reply");

    let resp_val: serde_json::Value = serde_json::from_slice(&resp.payload).unwrap();
    assert_eq!(resp_val["id"], "req_1");
    assert!(resp_val.get("result").is_some(), "Expected successful result");

    // 2. Send request with invalid token
    let req_invalid = serde_json::json!({
        "jsonrpc": "2.0",
        "id": "req_2",
        "method": "tasks/send",
        "params": {
            "message": {
                "metadata": {
                    "agent_jwt": "bad_token"
                }
            }
        }
    });

    let resp_err = client
        .request("a2a.v1.test.tasks.send".to_string(), serde_json::to_vec(&req_invalid).unwrap().into())
        .await
        .expect("Request should receive reply");

    let resp_err_val: serde_json::Value = serde_json::from_slice(&resp_err.payload).unwrap();
    assert!(resp_err_val.get("error").is_some(), "Expected auth error reply");

    serve_handle.abort();
}
