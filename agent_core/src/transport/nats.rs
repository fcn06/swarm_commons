use std::sync::Arc;
use a2a_rs::port::authenticator::{AuthContext, Authenticator};
use a2a_rs::services::server::AsyncA2ARequestProcessor;
use futures::StreamExt;
use serde_json::Value;
use tokio::sync::Semaphore;

use crate::server::auth::SharedAuthenticator;

/// Configuration for NATS dispatch listener
#[derive(Debug, Clone)]
pub struct NatsDispatchConfig {
    /// Subject to subscribe to, e.g. "a2a.v1.myagent.tasks.send"
    pub subject: String,
    /// Queue group for horizontal scaling across instances
    pub queue_group: Option<String>,
    /// Maximum concurrent in-flight requests processed (backpressure semaphore)
    pub max_in_flight: usize,
}

impl Default for NatsDispatchConfig {
    fn default() -> Self {
        Self {
            subject: "a2a.v1.*.tasks.send".to_string(),
            queue_group: Some("agent_workers".to_string()),
            max_in_flight: 64,
        }
    }
}

/// Trait to extract credentials from raw NATS payload (e.g. metadata.agent_jwt)
pub trait CredentialExtractor: Send + Sync {
    fn extract(&self, raw: &[u8]) -> Option<String>;
}

/// Default extractor: looks inside JSON-RPC params.message.metadata.agent_jwt or session_jwt or authorization
#[derive(Debug, Default, Clone)]
pub struct DefaultCredentialExtractor;

impl CredentialExtractor for DefaultCredentialExtractor {
    fn extract(&self, raw: &[u8]) -> Option<String> {
        let v: Value = serde_json::from_slice(raw).ok()?;
        // Support either direct metadata or params.message.metadata
        let metadata = v
            .get("params")
            .and_then(|p| p.get("message"))
            .and_then(|m| m.get("metadata"))
            .or_else(|| v.get("metadata"))
            .and_then(|m| m.as_object())?;

        let req_ctx = agent_models::RequestContext::from_metadata(Some(metadata));
        req_ctx.credential
    }
}

/// Serve A2A requests over NATS in-process using AsyncA2ARequestProcessor.
///
/// Ensures full parity with HTTP:
/// If an authenticator is present, the message is authenticated before processing.
/// On auth failure, replies with a JSON-RPC error.
pub async fn serve_nats<P: AsyncA2ARequestProcessor + Clone + 'static>(
    client: async_nats::Client,
    cfg: NatsDispatchConfig,
    processor: P,
    authenticator: Option<SharedAuthenticator>,
    extractor: Arc<dyn CredentialExtractor>,
) -> anyhow::Result<()> {
    let mut subscriber = match cfg.queue_group {
        Some(ref qg) => client.queue_subscribe(cfg.subject.clone(), qg.clone()).await?,
        None => client.subscribe(cfg.subject.clone()).await?,
    };

    tracing::info!(
        subject = %cfg.subject,
        queue_group = ?cfg.queue_group,
        max_in_flight = cfg.max_in_flight,
        "📡 NATS A2A dispatcher listening"
    );

    let semaphore = Arc::new(Semaphore::new(cfg.max_in_flight));

    while let Some(msg) = subscriber.next().await {
        let reply_to = match msg.reply {
            Some(r) => r,
            None => {
                tracing::warn!("Dropping NATS message without reply subject");
                continue;
            }
        };

        let permit = match semaphore.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => break, // semaphore closed
        };

        let client = client.clone();
        let processor = processor.clone();
        let authenticator = authenticator.clone();
        let extractor = extractor.clone();

        tokio::spawn(async move {
            let _permit = permit;

            // 1. Authenticate if authenticator is configured
            if let Some(auth) = authenticator {
                let cred = extractor.extract(&msg.payload);
                match cred {
                    Some(token) => {
                        let auth_ctx = AuthContext::new("bearer".to_string(), token);
                        if let Err(e) = auth.authenticate(&auth_ctx).await {
                            tracing::warn!("NATS request authentication failed: {}", e);
                            let rpc_err = serde_json::json!({
                                "jsonrpc": "2.0",
                                "error": {
                                    "code": -32000,
                                    "message": format!("Unauthorized: {}", e)
                                },
                                "id": null
                            });
                            let _ = client.publish(reply_to, serde_json::to_vec(&rpc_err).unwrap_or_default().into()).await;
                            return;
                        }
                    }
                    None => {
                        tracing::warn!("NATS request missing credential while auth is required");
                        let rpc_err = serde_json::json!({
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32000,
                                "message": "Unauthorized: missing credentials"
                            },
                            "id": null
                        });
                        let _ = client.publish(reply_to, serde_json::to_vec(&rpc_err).unwrap_or_default().into()).await;
                        return;
                    }
                }
            }

            // 2. Dispatch raw request in-process
            let raw_str = match std::str::from_utf8(&msg.payload) {
                Ok(s) => s,
                Err(e) => {
                    let rpc_err = serde_json::json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": format!("Parse error: invalid utf8: {}", e)
                        },
                        "id": null
                    });
                    let _ = client.publish(reply_to, serde_json::to_vec(&rpc_err).unwrap_or_default().into()).await;
                    return;
                }
            };

            match processor.process_raw_request(raw_str).await {
                Ok(resp_str) => {
                    let _ = client.publish(reply_to, resp_str.into_bytes().into()).await;
                }
                Err(e) => {
                    tracing::error!("Processor error: {}", e);
                    let rpc_err = serde_json::json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": format!("Internal error: {}", e)
                        },
                        "id": null
                    });
                    let _ = client.publish(reply_to, serde_json::to_vec(&rpc_err).unwrap_or_default().into()).await;
                }
            }
        });
    }

    Ok(())
}
