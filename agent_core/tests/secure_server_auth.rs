use std::sync::Arc;
use async_trait::async_trait;
use a2a_rs::domain::core::agent::SecurityScheme;
use a2a_rs::domain::A2AError;
use a2a_rs::port::authenticator::{AuthContext, AuthPrincipal, Authenticator};
use agent_core::server::auth::{OAuth2JwtAuthenticator, SharedAuthenticator};
use agent_core::server::secure_agent_server::{
    parse_bind_address, register_with_discovery_service, DiscoveryRetryPolicy,
};
use agent_models::registry::registry_models::{AgentDefinition, TaskDefinition, ToolDefinition};
use agent_core::business_logic::services::DiscoveryService;

// Mock authenticator that rejects any credential other than "allowed_token"
#[derive(Clone)]
struct MockRejectAuthenticator {
    scheme: SecurityScheme,
}

impl MockRejectAuthenticator {
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
impl Authenticator for MockRejectAuthenticator {
    async fn authenticate(&self, context: &AuthContext) -> Result<AuthPrincipal, A2AError> {
        if context.credential == "allowed_token" {
            Ok(AuthPrincipal::new("user_123".to_string(), "bearer".to_string()))
        } else {
            Err(A2AError::Internal("Unauthorized token".to_string()))
        }
    }

    fn security_scheme(&self) -> &SecurityScheme {
        &self.scheme
    }

    fn validate_context(&self, _context: &AuthContext) -> Result<(), A2AError> {
        Ok(())
    }
}

// Mock Discovery Service counting register_agent calls
struct MockDiscoveryService {
    calls: std::sync::atomic::AtomicUsize,
    succeed_on_attempt: usize,
}

#[async_trait]
impl DiscoveryService for MockDiscoveryService {
    async fn register_agent(&self, _agent_def: &AgentDefinition) -> anyhow::Result<()> {
        let current = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        if current >= self.succeed_on_attempt {
            Ok(())
        } else {
            anyhow::bail!("Discovery temporary error (attempt {})", current);
        }
    }
    async fn unregister_agent(&self, _agent_def: &AgentDefinition) -> anyhow::Result<()> { Ok(()) }
    async fn get_agent_address(&self, _agent_id: String) -> anyhow::Result<Option<String>> { Ok(None) }
    async fn discover_agents(&self) -> anyhow::Result<Vec<AgentDefinition>> { Ok(vec![]) }
    async fn register_task(&self, _task_def: &TaskDefinition) -> anyhow::Result<()> { Ok(()) }
    async fn list_tasks(&self) -> anyhow::Result<Vec<TaskDefinition>> { Ok(vec![]) }
    async fn register_tool(&self, _tool_def: &ToolDefinition) -> anyhow::Result<()> { Ok(()) }
    async fn list_tools(&self) -> anyhow::Result<Vec<ToolDefinition>> { Ok(vec![]) }
    async fn list_available_resources(&self) -> anyhow::Result<String> { Ok("".to_string()) }
}

#[tokio::test]
async fn test_shared_authenticator_injection() {
    let mock_auth = MockRejectAuthenticator::new();
    let shared = SharedAuthenticator::new(mock_auth);

    // Test rejection
    let bad_ctx = AuthContext::new("bearer".to_string(), "bad_token".to_string());
    let res_bad = shared.authenticate(&bad_ctx).await;
    assert!(res_bad.is_err());

    // Test acceptance
    let good_ctx = AuthContext::new("bearer".to_string(), "allowed_token".to_string());
    let res_good = shared.authenticate(&good_ctx).await.unwrap();
    assert_eq!(res_good.id, "user_123");
}

#[test]
fn test_jwt_validation_hardened() {
    // Missing audience or issuer must fail constructor
    assert!(OAuth2JwtAuthenticator::try_new("secret", "", "issuer").is_err());
    assert!(OAuth2JwtAuthenticator::try_new("secret", "aud", "").is_err());
    assert!(OAuth2JwtAuthenticator::try_new("", "aud", "issuer").is_err());

    let auth = OAuth2JwtAuthenticator::try_new("super_secret_test_key_12345", "test_aud", "test_iss").unwrap();

    // Verify negative cases with jsonwebtoken
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // 1. Expired token
        let my_claims = serde_json::json!({
            "sub": "user1",
            "aud": "test_aud",
            "iss": "test_iss",
            "exp": 1000000000usize, // in the past
        });
        let key = jsonwebtoken::EncodingKey::from_secret("super_secret_test_key_12345".as_bytes());
        let token = jsonwebtoken::encode(&jsonwebtoken::Header::default(), &my_claims, &key).unwrap();

        let ctx = AuthContext::new("bearer".to_string(), token);
        let err = auth.authenticate(&ctx).await;
        assert!(err.is_err(), "Expired token must be rejected");

        // 2. Wrong audience
        let my_claims_aud = serde_json::json!({
            "sub": "user1",
            "aud": "wrong_aud",
            "iss": "test_iss",
            "exp": 2500000000usize,
        });
        let token_aud = jsonwebtoken::encode(&jsonwebtoken::Header::default(), &my_claims_aud, &key).unwrap();
        let ctx_aud = AuthContext::new("bearer".to_string(), token_aud);
        let err_aud = auth.authenticate(&ctx_aud).await;
        assert!(err_aud.is_err(), "Wrong audience must be rejected");

        // 3. Wrong issuer
        let my_claims_iss = serde_json::json!({
            "sub": "user1",
            "aud": "test_aud",
            "iss": "wrong_iss",
            "exp": 2500000000usize,
        });
        let token_iss = jsonwebtoken::encode(&jsonwebtoken::Header::default(), &my_claims_iss, &key).unwrap();
        let ctx_iss = AuthContext::new("bearer".to_string(), token_iss);
        let err_iss = auth.authenticate(&ctx_iss).await;
        assert!(err_iss.is_err(), "Wrong issuer must be rejected");
    });
}

#[tokio::test]
async fn test_discovery_retry() {
    let mock_ds = Arc::new(MockDiscoveryService {
        calls: std::sync::atomic::AtomicUsize::new(0),
        succeed_on_attempt: 2,
    });

    let def = AgentDefinition {
        id: "a1".to_string(),
        name: "TestAgent".to_string(),
        description: "Test".to_string(),
        agent_endpoint: "http://127.0.0.1:8080".to_string(),
        skills: vec![],
    };

    let policy = DiscoveryRetryPolicy {
        max_retries: 3,
        initial_delay: std::time::Duration::from_millis(10),
        max_delay: std::time::Duration::from_millis(50),
    };

    let ds_opt: Option<Arc<dyn DiscoveryService>> = Some(mock_ds.clone());
    let res = register_with_discovery_service(&ds_opt, &def, policy).await;
    assert!(res.is_ok());
    assert_eq!(mock_ds.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[test]
fn test_bind_address_parsing() {
    assert_eq!(parse_bind_address("http://127.0.0.1:9090").unwrap(), "127.0.0.1:9090");
    assert_eq!(parse_bind_address("https://example.com:8443/agent").unwrap(), "example.com:8443");
    assert_eq!(parse_bind_address("0.0.0.0:8000").unwrap(), "0.0.0.0:8000");
}
