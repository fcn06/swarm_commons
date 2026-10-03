use std::sync::Arc;
use std::time::Duration;
use async_trait::async_trait;
use a2a_rs::domain::{Message, TaskState};
use a2a_rs::port::AsyncMessageHandler;
use agent_core::business_logic::agent::Agent;
use agent_core::business_logic::mcp_runtime::McpRuntimeDetails;
use agent_core::business_logic::services::{DiscoveryService, EvaluationService, MemoryService, WorkflowServiceApi};
use agent_core::server::agent_handler::AgentHandler;
use agent_core::session::{SessionStore, SessionStoreApi, TenantThreadResolver};
use agent_models::agent_request::AgentRequest;
use agent_models::execution::execution_result::ExecutionResult;
use configuration::AgentConfig;
use serde_json::Value;

#[derive(Clone)]
struct SlowAgent;

#[async_trait]
impl Agent for SlowAgent {
    async fn new(
        _agent_config: AgentConfig,
        _agent_api_key: String,
        _mcp_runtime_details: Option<McpRuntimeDetails>,
        _evaluation_service: Option<Arc<dyn EvaluationService>>,
        _memory_service: Option<Arc<dyn MemoryService>>,
        _discovery_service: Option<Arc<dyn DiscoveryService>>,
        _workflow_service: Option<Arc<dyn WorkflowServiceApi>>,
    ) -> anyhow::Result<Self> {
        Ok(Self)
    }

    async fn handle_request(&self, request: AgentRequest) -> anyhow::Result<ExecutionResult> {
        let q = request.user_query();
        if q == "sleep" {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        Ok(ExecutionResult {
            request_id: "req_1".to_string(),
            conversation_id: "conv_1".to_string(),
            success: true,
            output: Value::String(format!("Echo: {}", q)),
        })
    }
}

#[tokio::test]
async fn test_agent_handler_concurrency() {
    let handler = Arc::new(AgentHandler::new(SlowAgent));

    let start = std::time::Instant::now();
    let mut handles = Vec::new();

    for i in 0..4 {
        let h = Arc::clone(&handler);
        handles.push(tokio::spawn(async move {
            let msg = Message::user_text("sleep".to_string(), format!("msg_{}", i));
            let task_id = format!("task_{}", i);
            h.process_message(&task_id, &msg, Some(&format!("sess_{}", i)))
                .await
                .expect("process_message should succeed")
        }));
    }

    for h in handles {
        let task = h.await.expect("task join failed");
        assert_eq!(task.status.state, TaskState::Completed);
    }

    let elapsed = start.elapsed();
    // 4 concurrent tasks of 500ms must finish well under 1s (e.g. around 500-600ms), NOT 2.0s
    assert!(
        elapsed < Duration::from_millis(900),
        "Expected elapsed under 900ms, got {:?}",
        elapsed
    );
}

#[tokio::test]
async fn test_agent_handler_assistant_append_and_tenant_isolation() {
    let store = Arc::new(SessionStore::new());
    let resolver = Arc::new(TenantThreadResolver);
    let handler = AgentHandler::builder(SlowAgent)
        .session_store(store.clone() as Arc<dyn SessionStoreApi>)
        .session_key_resolver(resolver)
        .build();

    // Turn 1 for Tenant A, Thread 1
    let mut meta_a = serde_json::Map::new();
    meta_a.insert("tenant_id".into(), Value::String("tenantA".into()));
    meta_a.insert("thread_id".into(), Value::String("thread1".into()));

    let mut msg1 = Message::user_text("Hello from Tenant A".to_string(), "msg_a1".to_string());
    msg1.metadata = Some(meta_a);

    handler
        .process_message("task_a1", &msg1, None)
        .await
        .unwrap();

    // Turn 1 for Tenant B, Thread 1 (same thread id, different tenant)
    let mut meta_b = serde_json::Map::new();
    meta_b.insert("tenant_id".into(), Value::String("tenantB".into()));
    meta_b.insert("thread_id".into(), Value::String("thread1".into()));

    let mut msg2 = Message::user_text("Hello from Tenant B".to_string(), "msg_b1".to_string());
    msg2.metadata = Some(meta_b);

    handler
        .process_message("task_b1", &msg2, None)
        .await
        .unwrap();

    // Inspect histories
    let history_a = store.get_history("tenantA_thread1").await;
    let history_b = store.get_history("tenantB_thread1").await;

    // Both should have 2 items: user question + assistant response
    assert_eq!(history_a.len(), 2, "Tenant A history should contain user + assistant items");
    assert_eq!(history_b.len(), 2, "Tenant B history should contain user + assistant items");

    // Verify isolation
    let q_a = match &history_a[0] {
        agent_models::response_item::ResponseItem::Message { content, .. } => {
            match &content[0] {
                agent_models::response_item::ContentPart::Text { text } => text.clone(),
                _ => String::new(),
            }
        }
        _ => String::new(),
    };
    assert_eq!(q_a, "Hello from Tenant A");

    let q_b = match &history_b[0] {
        agent_models::response_item::ResponseItem::Message { content, .. } => {
            match &content[0] {
                agent_models::response_item::ContentPart::Text { text } => text.clone(),
                _ => String::new(),
            }
        }
        _ => String::new(),
    };
    assert_eq!(q_b, "Hello from Tenant B");
}

#[tokio::test]
async fn test_ephemeral_turn_no_default_session() {
    let store = Arc::new(SessionStore::new());
    let handler = AgentHandler::builder(SlowAgent)
        .session_store(store.clone() as Arc<dyn SessionStoreApi>)
        .build();

    let msg = Message::user_text("Ephemeral turn".to_string(), "msg_e1".to_string());
    let task = handler
        .process_message("task_ephemeral", &msg, None)
        .await
        .unwrap();

    assert_eq!(task.status.state, TaskState::Completed);

    // Verify "default_session" does not exist in store!
    let history = store.get_history("default_session").await;
    assert_eq!(history.len(), 0, "No default_session should be created");
}
