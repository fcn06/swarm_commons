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
    #[allow(dead_code)]
    fn create_in_memory_storage_with_push_notification(&self) -> InMemoryTaskStorage {
        tracing::info!("Using in-memory storage with push notification support");
        let push_sender = HttpPushNotificationSender::new()
            .with_timeout(30)
            .with_max_retries(3);
        InMemoryTaskStorage::with_push_sender(push_sender)
    }

    async fn register_with_discovery_service(&self, agent_definition: &AgentDefinition) -> Result<()> {
        let max_retries = 3;
        let mut retries = 0;
        let mut delay = std::time::Duration::from_millis(1000);

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
                            tracing::warn!("Failed to register with discovery service, attempt {}/{}. Error: {}. Retrying in {:?}...", retries, max_retries, e, delay);
                            tokio::time::sleep(delay).await;
                            delay = std::cmp::min(delay * 2, std::time::Duration::from_secs(30));
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

                let authenticator = ApiKeyAuthenticator::new(keys.clone(), location, name);
                let server =
                    HttpServer::with_auth(processor, agent_info, bind_address, authenticator);
                server
                    .start()
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            AuthConfig::OAuth2Jwt { secret, audience, issuer } => {
                println!("🔐 Authentication: OAuth2 JWT Bearer Token validation");
                let authenticator = OAuth2JwtAuthenticator::new(secret, audience.clone(), issuer.clone());
                let server = HttpServer::with_auth(processor, agent_info, bind_address, authenticator);
                server
                    .start()
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
        }


    }
}

/// Dynamic API Key authenticator wrapper for AXUM HTTP server
#[derive(Clone)]
pub struct ApiKeyAuthenticator {
    keys: Vec<String>,
    scheme: a2a_rs::domain::core::agent::SecurityScheme,
}

impl ApiKeyAuthenticator {
    pub fn new(keys: Vec<String>, location: &str, name: &str) -> Self {
        Self {
            keys,
            scheme: a2a_rs::domain::core::agent::SecurityScheme::ApiKey {
                name: name.to_string(),
                location: location.to_string(),
                description: Some("API Key Authentication".to_string()),
            },
        }
    }
}

#[async_trait::async_trait]
impl a2a_rs::port::authenticator::Authenticator for ApiKeyAuthenticator {
    async fn authenticate(
        &self,
        context: &a2a_rs::port::authenticator::AuthContext,
    ) -> Result<a2a_rs::port::authenticator::AuthPrincipal, a2a_rs::domain::A2AError> {
        self.validate_context(context)?;

        if self.keys.iter().any(|k| k == &context.credential) {
            let key_suffix = if context.credential.len() >= 4 {
                &context.credential[context.credential.len() - 4..]
            } else {
                &context.credential
            };
            Ok(a2a_rs::port::authenticator::AuthPrincipal::new(
                format!("apikey_principal_...{}", key_suffix),
                "apikey".to_string(),
            ))
        } else {
            Err(a2a_rs::domain::A2AError::Internal(
                "API key authentication failed: invalid API key".to_string(),
            ))
        }
    }

    fn security_scheme(&self) -> &a2a_rs::domain::core::agent::SecurityScheme {
        &self.scheme
    }

    fn validate_context(
        &self,
        context: &a2a_rs::port::authenticator::AuthContext,
    ) -> Result<(), a2a_rs::domain::A2AError> {
        if context.scheme_type != "apikey" && context.scheme_type != "bearer" && context.scheme_type != "header" {
            return Err(a2a_rs::domain::A2AError::Internal(format!(
                "Invalid authentication scheme for API key: expected 'apikey', 'bearer', or 'header', got '{}'",
                context.scheme_type
            )));
        }
        Ok(())
    }
}

/// Dynamic OAuth2 JWT authenticator wrapper for AXUM HTTP server
#[derive(Clone)]
pub struct OAuth2JwtAuthenticator {
    secret: String,
    audience: String,
    issuer: String,
    scheme: a2a_rs::domain::core::agent::SecurityScheme,
}

#[derive(Debug, Serialize, Deserialize)]
struct JwtClaims {
    #[serde(default)]
    pub sub: Option<String>,
    #[serde(default)]
    pub aud: Option<serde_json::Value>,
    #[serde(default)]
    pub iss: Option<String>,
    #[serde(default)]
    pub exp: Option<usize>,
    #[serde(default)]
    pub nbf: Option<usize>,
    #[serde(default)]
    pub iat: Option<usize>,
    #[serde(default)]
    pub jti: Option<String>,
    #[serde(default)]
    pub tenant_id: Option<String>,
}

impl OAuth2JwtAuthenticator {
    pub fn new(secret: &str, audience: String, issuer: String) -> Self {
        Self {
            secret: secret.to_string(),
            audience,
            issuer,
            scheme: a2a_rs::domain::core::agent::SecurityScheme::Http {
                scheme: "bearer".to_string(),
                bearer_format: Some("JWT".to_string()),
                description: Some("OAuth2 JWT Bearer Token".to_string()),
            },
        }
    }
}

#[async_trait::async_trait]
impl a2a_rs::port::authenticator::Authenticator for OAuth2JwtAuthenticator {
    async fn authenticate(
        &self,
        context: &a2a_rs::port::authenticator::AuthContext,
    ) -> Result<a2a_rs::port::authenticator::AuthPrincipal, a2a_rs::domain::A2AError> {
        self.validate_context(context)?;

        let mut validation = jsonwebtoken::Validation::new(jsonwebtoken::Algorithm::HS256);
        if !self.audience.is_empty() {
            validation.set_audience(&[&self.audience]);
        } else {
            validation.validate_aud = false;
        }
        if !self.issuer.is_empty() {
            validation.set_issuer(&[&self.issuer]);
        }

        let decoding_key = jsonwebtoken::DecodingKey::from_secret(self.secret.as_bytes());
        match jsonwebtoken::decode::<JwtClaims>(&context.credential, &decoding_key, &validation) {
            Ok(token_data) => {
                let principal_id = token_data.claims.tenant_id
                    .or(token_data.claims.sub)
                    .unwrap_or_else(|| "anonymous".to_string());
                Ok(a2a_rs::port::authenticator::AuthPrincipal::new(
                    principal_id,
                    "bearer".to_string(),
                ))
            }
            Err(e) => Err(a2a_rs::domain::A2AError::Internal(format!(
                "OAuth2 JWT verification failed: {}",
                e
            ))),
        }
    }

    fn security_scheme(&self) -> &a2a_rs::domain::core::agent::SecurityScheme {
        &self.scheme
    }

    fn validate_context(
        &self,
        context: &a2a_rs::port::authenticator::AuthContext,
    ) -> Result<(), a2a_rs::domain::A2AError> {
        if context.scheme_type != "bearer" {
            return Err(a2a_rs::domain::A2AError::Internal(format!(
                "Invalid authentication scheme: expected 'bearer', got '{}'",
                context.scheme_type
            )));
        }
        Ok(())
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
    /// OAuth2 JWT Bearer authentication
    OAuth2Jwt {
        secret: String,
        audience: String,
        issuer: String,
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
        // Check for JWT secret first
        if let Ok(secret) = env::var("AUTH_JWT_SECRET") {
            let audience = env::var("AUTH_JWT_AUDIENCE").unwrap_or_default();
            let issuer = env::var("AUTH_JWT_ISSUER").unwrap_or_default();
            return Self::OAuth2Jwt { secret, audience, issuer };
        }

        // Check for bearer tokens
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