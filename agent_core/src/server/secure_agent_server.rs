use std::sync::Arc;
use std::time::Duration;
use anyhow::Result;
use a2a_rs::adapter::{DefaultRequestProcessor, HttpServer, InMemoryTaskStorage, NoopPushNotificationSender, SimpleAgentInfo};
use agent_models::registry::registry_models::{AgentDefinition, AgentSkillDefinition};
use configuration::AgentConfig;
use uuid::Uuid;
use url::Url;

use crate::business_logic::agent::Agent;
use crate::business_logic::services::DiscoveryService;
use crate::server::agent_handler::AgentHandler;
use crate::server::auth::{AuthConfig, SharedAuthenticator};

/// Retry policy for registering an agent with the discovery service
#[derive(Debug, Clone)]
pub struct DiscoveryRetryPolicy {
    pub max_retries: usize,
    pub initial_delay: Duration,
    pub max_delay: Duration,
}

impl Default for DiscoveryRetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(30),
        }
    }
}

/// Free function to register an agent definition with a discovery service with retry policy
pub async fn register_with_discovery_service(
    discovery_service: &Option<Arc<dyn DiscoveryService>>,
    agent_definition: &AgentDefinition,
    retry_policy: DiscoveryRetryPolicy,
) -> Result<()> {
    if let Some(ds) = discovery_service {
        let mut retries = 0;
        let mut delay = retry_policy.initial_delay;

        loop {
            match ds.register_agent(agent_definition).await {
                Ok(_) => {
                    tracing::info!("Agent successfully registered with discovery service.");
                    break;
                }
                Err(e) => {
                    retries += 1;
                    if retries < retry_policy.max_retries {
                        tracing::warn!(
                            "Failed to register with discovery service, attempt {}/{}. Error: {}. Retrying in {:?}...",
                            retries, retry_policy.max_retries, e, delay
                        );
                        tokio::time::sleep(delay).await;
                        delay = std::cmp::min(delay * 2, retry_policy.max_delay);
                    } else {
                        tracing::error!(
                            "Failed to register with discovery service after {} attempts. Error: {}. Proceeding without discovery service registration.",
                            retry_policy.max_retries, e
                        );
                        return Ok(());
                    }
                }
            }
        }
    } else {
        tracing::warn!("Discovery service not configured. Skipping registration.");
    }
    Ok(())
}

/// Helper function to build SimpleAgentInfo from AgentConfig
pub fn build_agent_info(config: &AgentConfig) -> SimpleAgentInfo {
    let agent_http_endpoint = config.agent_http_endpoint().to_string();
    let mut info = SimpleAgentInfo::new(config.agent_name(), agent_http_endpoint)
        .with_description(config.agent_description())
        .with_streaming()
        .add_comprehensive_skill(
            config.agent_skill_id(),
            config.agent_skill_name(),
            Some(config.agent_skill_description()),
            Some(config.agent_tags()),
            Some(config.agent_examples()),
            Some(vec!["text".to_string(), "data".to_string()]),
            Some(vec!["text".to_string(), "data".to_string()]),
        );

    if let Some(doc_url) = config.agent_doc_url() {
        info = info.with_documentation_url(doc_url);
    }
    info
}

/// Helper function to build AgentDefinition from AgentConfig
pub fn build_agent_definition(config: &AgentConfig) -> AgentDefinition {
    let agent_http_endpoint = config.agent_http_endpoint().to_string();
    AgentDefinition {
        id: Uuid::new_v4().to_string(),
        name: config.agent_name(),
        description: config.agent_description(),
        agent_endpoint: agent_http_endpoint,
        skills: vec![AgentSkillDefinition {
            name: config.agent_skill_name(),
            description: config.agent_skill_description(),
            parameters: serde_json::Value::Null,
            output: serde_json::Value::Null,
        }],
    }
}

/// Helper function to parse host and port from endpoint URL
pub fn parse_bind_address(endpoint: &str) -> Result<String, anyhow::Error> {
    if let Ok(url) = Url::parse(endpoint) {
        let host = url.host_str().unwrap_or("0.0.0.0");
        let port = url.port().unwrap_or(80);
        Ok(format!("{}:{}", host, port))
    } else {
        // Fallback if not a full URL: strip http:// or https:// if present
        let stripped = endpoint
            .trim_start_matches("http://")
            .trim_start_matches("https://");
        let host_port = stripped.split('/').next().unwrap_or(stripped);
        if host_port.is_empty() {
            anyhow::bail!("Invalid agent http endpoint: {}", endpoint);
        }
        Ok(host_port.to_string())
    }
}

/// Builder for SecureAgentServer allowing extension points and injection
pub struct SecureAgentServerBuilder<T: Agent> {
    config: AgentConfig,
    agent: T,
    auth: Option<SharedAuthenticator>,
    discovery_service: Option<Arc<dyn DiscoveryService>>,
    discovery_retry_policy: DiscoveryRetryPolicy,
    handler: Option<AgentHandler<T>>,
    #[cfg(feature = "nats")]
    nats_config: Option<(
        async_nats::Client,
        crate::transport::nats::NatsDispatchConfig,
        Arc<dyn crate::transport::nats::CredentialExtractor>,
    )>,
}

impl<T: Agent> SecureAgentServerBuilder<T> {
    pub fn new(config: AgentConfig, agent: T) -> Self {
        Self {
            config,
            agent,
            auth: None,
            discovery_service: None,
            discovery_retry_policy: DiscoveryRetryPolicy::default(),
            handler: None,
            #[cfg(feature = "nats")]
            nats_config: None,
        }
    }

    pub fn authenticator(mut self, auth: SharedAuthenticator) -> Self {
        self.auth = Some(auth);
        self
    }

    pub fn maybe_authenticator(mut self, auth: Option<SharedAuthenticator>) -> Self {
        self.auth = auth;
        self
    }

    pub fn auth_config(mut self, auth_config: AuthConfig) -> Result<Self, anyhow::Error> {
        self.auth = auth_config.to_shared_authenticator()?;
        Ok(self)
    }

    pub fn discovery(mut self, ds: Option<Arc<dyn DiscoveryService>>) -> Self {
        self.discovery_service = ds;
        self
    }

    pub fn discovery_retry_policy(mut self, policy: DiscoveryRetryPolicy) -> Self {
        self.discovery_retry_policy = policy;
        self
    }

    pub fn handler(mut self, handler: AgentHandler<T>) -> Self {
        self.handler = Some(handler);
        self
    }

    #[cfg(feature = "nats")]
    pub fn nats(
        mut self,
        client: async_nats::Client,
        cfg: crate::transport::nats::NatsDispatchConfig,
    ) -> Self {
        self.nats_config = Some((client, cfg, Arc::new(crate::transport::nats::DefaultCredentialExtractor)));
        self
    }

    #[cfg(feature = "nats")]
    pub fn nats_with_extractor(
        mut self,
        client: async_nats::Client,
        cfg: crate::transport::nats::NatsDispatchConfig,
        extractor: Arc<dyn crate::transport::nats::CredentialExtractor>,
    ) -> Self {
        self.nats_config = Some((client, cfg, extractor));
        self
    }

    pub fn build(self) -> SecureAgentServer<T> {
        let handler = self.handler.unwrap_or_else(|| {
            let storage = InMemoryTaskStorage::with_push_sender(NoopPushNotificationSender);
            AgentHandler::with_storage(self.agent.clone(), storage)
        });

        SecureAgentServer {
            config: self.config,
            agent: self.agent,
            auth: self.auth,
            discovery_service: self.discovery_service,
            discovery_retry_policy: self.discovery_retry_policy,
            handler,
            #[cfg(feature = "nats")]
            nats_config: self.nats_config,
        }
    }
}

pub struct SecureAgentServer<T: Agent> {
    config: AgentConfig,
    #[allow(dead_code)]
    agent: T,
    auth: Option<SharedAuthenticator>,
    discovery_service: Option<Arc<dyn DiscoveryService>>,
    discovery_retry_policy: DiscoveryRetryPolicy,
    handler: AgentHandler<T>,
    #[cfg(feature = "nats")]
    nats_config: Option<(
        async_nats::Client,
        crate::transport::nats::NatsDispatchConfig,
        Arc<dyn crate::transport::nats::CredentialExtractor>,
    )>,
}

impl<T: Agent> SecureAgentServer<T> {
    pub fn builder(config: AgentConfig, agent: T) -> SecureAgentServerBuilder<T> {
        SecureAgentServerBuilder::new(config, agent)
    }

    pub async fn new(
        agent_config: AgentConfig,
        agent: T,
        auth: AuthConfig,
        discovery_service: Option<Arc<dyn DiscoveryService>>,
    ) -> anyhow::Result<Self> {
        let auth_authenticator = auth.to_shared_authenticator()?;
        Ok(Self::builder(agent_config, agent)
            .maybe_authenticator(auth_authenticator)
            .discovery(discovery_service)
            .build())
    }

    pub async fn start_http(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.start_http_internal(None).await
    }

    pub async fn start_http_with_shutdown(
        &self,
        shutdown_signal: impl std::future::Future<Output = ()> + Send + 'static,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.start_http_internal(Some(Box::pin(shutdown_signal))).await
    }

    async fn start_http_internal(
        &self,
        _shutdown_signal: Option<std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let agent_http_endpoint = self.config.agent_http_endpoint().to_string();
        let bind_address = parse_bind_address(&agent_http_endpoint)?;

        let agent_info = build_agent_info(&self.config);
        let agent_definition = build_agent_definition(&self.config);

        if let Some(true) = self.config.agent_discoverable() {
            register_with_discovery_service(
                &self.discovery_service,
                &agent_definition,
                self.discovery_retry_policy.clone(),
            )
            .await?;
        }

        println!(
            "🌐 Starting HTTP a2a agent server {} on {}",
            self.config.agent_name(), self.config.agent_http_endpoint()
        );
        println!(
            "📋 Agent card: {}/agent-card",
            self.config.agent_http_endpoint(),
        );
        println!(
            "🛠️  Skills: {}/skills",
            self.config.agent_http_endpoint()
        );
        println!("💾 Storage: In-memory (non-persistent)");

        let processor = DefaultRequestProcessor::with_handler(self.handler.clone(), agent_info.clone());

        #[cfg(feature = "nats")]
        if let Some((client, nats_cfg, extractor)) = &self.nats_config {
            let nats_proc = processor.clone();
            let nats_auth = self.auth.clone();
            let nats_client = client.clone();
            let nats_cfg = nats_cfg.clone();
            let nats_extractor = extractor.clone();
            tokio::spawn(async move {
                if let Err(e) = crate::transport::nats::serve_nats(
                    nats_client,
                    nats_cfg,
                    nats_proc,
                    nats_auth,
                    nats_extractor,
                )
                .await
                {
                    tracing::error!("NATS dispatcher error: {}", e);
                }
            });
        }

        match &self.auth {
            None => {
                println!("🔓 Authentication: None (public access)");
                let server = HttpServer::new(processor, agent_info, bind_address);
                server.start().await.map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            Some(authenticator) => {
                println!("🔐 Authentication: Active (SharedAuthenticator)");
                let server = HttpServer::with_auth(processor, agent_info, bind_address, authenticator.clone());
                server.start().await.map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
        }
    }
}