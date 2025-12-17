use a2a_rs::adapter::{
    BearerTokenAuthenticator, DefaultRequestProcessor,  HttpServer,
    InMemoryTaskStorage, SimpleAgentInfo, 
    HttpPushNotificationSender,NoopPushNotificationSender,
};

//use a2a_rs::port::{AsyncNotificationManager, AsyncStreamingHandler, AsyncTaskManager};


use serde::{Serialize,Deserialize};

use crate::business_logic::agent::{Agent};

use configuration::AgentConfig;

use crate::server::agent_handler::AgentHandler;
use std::sync::Arc;
use crate::business_logic::services::DiscoveryService;

use anyhow::Result;

use uuid::Uuid;
use std::env;

use agent_models::registry::registry_models::{AgentDefinition,AgentSkillDefinition};


pub struct SecureAgentServer<T:Agent> {
    config: AgentConfig,
    agent:T,
    auth: AuthConfig,
    discovery_service: Option<Arc<dyn DiscoveryService>>,
}

impl<T:Agent> SecureAgentServer<T> {
    pub async fn new(agent_config: AgentConfig, agent: T, auth:AuthConfig,discovery_service: Option<Arc<dyn DiscoveryService>>) -> anyhow::Result<Self> {
        //Ok(Self { config:agent_config,agent:agent,auth:AuthConfig::default(),discovery_service:discovery_service })
        Ok(Self { config:agent_config,agent:agent,auth:auth,discovery_service:discovery_service })
    }

    /// Create in-memory storage without push notification
    fn create_in_memory_storage(&self) -> InMemoryTaskStorage {
        tracing::info!("Using in-memory storage");
        let push_sender = NoopPushNotificationSender;
        InMemoryTaskStorage::with_push_sender(push_sender)
    }

    /// Create in-memory storage with push notification
    fn create_in_memory_storage_with_push_notification(&self) -> InMemoryTaskStorage {
        tracing::info!("Using in-memory storage with push notification support");
        let push_sender = HttpPushNotificationSender::new()
            .with_timeout(30)
            .with_max_retries(3);
        InMemoryTaskStorage::with_push_sender(push_sender)
    }

    async fn register_with_discovery_service(&self, agent_definition: &AgentDefinition) -> Result<()> {
        let max_retries = 2;
        let mut retries = 0;
        let mut delay = 1; // seconds

        if let Some(ds) = &self.discovery_service {
            loop {
                let registration_result = ds.register_agent(&agent_definition).await;

                match registration_result {
                    Ok(_) => {
                        tracing::info!("Agent successfully registered with discovery service.");
                        break;
                    },
                    Err(e) => {
                        retries += 1;
                        if retries < max_retries {
                            tracing::warn!("Failed to register with discovery service, attempt {}/{}. Error: {}. Retrying in {} seconds...", retries, max_retries, e, delay);
                            tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                            delay *= 2; // Exponential backoff
                        } else {
                            tracing::error!("Failed to register with discovery service after {} attempts. Error: {}. Proceeding without discovery service registration.", max_retries, e);
                            // Allow the agent to start even if registration fails
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

    pub async fn start_http(&self) -> Result<(), Box<dyn std::error::Error>> {
        
        let storage = self.create_in_memory_storage();

        let message_handler = AgentHandler::<T>::with_storage(self.agent.clone(),storage.clone());

        let agent_http_endpoint= format!("{}", self.config.agent_http_endpoint());
        let _agent_ws_endpoint= format!("{}", self.config.agent_ws_endpoint());

        // We should remove that part
        let simple_agent_info = SimpleAgentInfo::new(
            self.config.agent_name(),
            agent_http_endpoint.clone(),
        );

        let processor = DefaultRequestProcessor::with_handler(message_handler, simple_agent_info);

        
        let agent_info = SimpleAgentInfo::new(
            self.config.agent_name(),
            agent_http_endpoint.clone(),
        )
        .with_description(self.config.agent_description())
        .with_documentation_url(self.config.agent_doc_url().expect("NO DOC URL PROVIDED IN CONFIG"))
        .with_streaming()
        .add_comprehensive_skill(
            self.config.agent_skill_id(),
            self.config.agent_skill_name(),
            Some(self.config.agent_skill_description()),
            Some(self.config.agent_tags()),
            Some(self.config.agent_examples()),
            Some(vec!["text".to_string(), "data".to_string()]),
            Some(vec!["text".to_string(), "data".to_string()]),
        );

        
        let agent_definition=AgentDefinition{
            id:Uuid::new_v4().to_string(),
            name:self.config.agent_name(),
            description:self.config.agent_description(),
            agent_endpoint:  agent_http_endpoint.clone(),
            skills:vec![AgentSkillDefinition{
                name:self.config.agent_skill_name(),
                description:self.config.agent_skill_description(),
                parameters:serde_json::Value::Null,
                output:serde_json::Value::Null,
            }]
        };


        if let Some(true) = self.config.agent_discoverable() {
            self.register_with_discovery_service(&agent_definition).await?;
        }

        // bind address is on format  0.0.0.0:0000
        let bind_address = agent_http_endpoint.clone().replace("http://","");

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
        
        /* 
        println!("🔓 Authentication: None (public access)");
        let server = HttpServer::new(processor, agent_info, bind_address);
        server
            .start()
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        */

        match &self.auth {
            AuthConfig::None => {
                println!("🔓 Authentication: None (public access)");

                // Create server without authentication
                let server = HttpServer::new(processor, agent_info, bind_address);
                server
                    .start()
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            AuthConfig::BearerToken { tokens, format } => {
                println!(
                    "🔐 Authentication: Bearer token ({} token(s){})",
                    tokens.len(),
                    format
                        .as_ref()
                        .map(|f| format!(", format: {}", f))
                        .unwrap_or_default()
                );

                let authenticator = BearerTokenAuthenticator::new(tokens.clone());
                let server =
                    HttpServer::with_auth(processor, agent_info, bind_address, authenticator);
                server
                    .start()
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            AuthConfig::ApiKey {
                keys,
                location,
                name,
            } => {
                println!(
                    "🔐 Authentication: API key ({} {}, {} key(s))",
                    location,
                    name,
                    keys.len()
                );
                println!("⚠️  API key authentication not yet supported, using no authentication");

                // Create server without authentication
                let server = HttpServer::new(processor, agent_info, bind_address);
                server
                    .start()
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
        }


    }
}



#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AuthConfig {
    /// No authentication (default for development)
    None,
    /// Bearer token authentication
    BearerToken {
        /// List of valid tokens
        tokens: Vec<String>,
        /// Optional bearer format description (e.g., "JWT")
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
    /// API Key authentication
    ApiKey {
        /// Valid API keys
        keys: Vec<String>,
        /// Location of the API key: "header", "query", or "cookie"
        #[serde(default = "default_api_key_location")]
        location: String,
        /// Name of the header/query param/cookie
        #[serde(default = "default_api_key_name")]
        name: String,
    },
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self::None
    }
}

impl AuthConfig {
    /// Create auth config from environment variables
    pub fn from_env() -> Self {
        // Check for bearer tokens first
        if let Ok(tokens_str) = env::var("AUTH_BEARER_TOKENS") {
            let tokens: Vec<String> = tokens_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            if !tokens.is_empty() {
                return Self::BearerToken {
                    tokens,
                    format: env::var("AUTH_BEARER_FORMAT").ok(),
                };
            }
        }

        // Check for API keys
        if let Ok(keys_str) = env::var("AUTH_API_KEYS") {
            let keys: Vec<String> = keys_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            if !keys.is_empty() {
                return Self::ApiKey {
                    keys,
                    location: env::var("AUTH_API_KEY_LOCATION")
                        .unwrap_or_else(|_| default_api_key_location()),
                    name: env::var("AUTH_API_KEY_NAME").unwrap_or_else(|_| default_api_key_name()),
                };
            }
        }

        // Default to no authentication
        Self::None
    }
}

fn default_api_key_location() -> String {
    "header".to_string()
}

fn default_api_key_name() -> String {
    "X-API-Key".to_string()
}