use std::sync::Arc;
use a2a_rs::domain::A2AError;
use a2a_rs::port::authenticator::{AuthContext, AuthPrincipal, Authenticator};
use a2a_rs::domain::core::agent::SecurityScheme;
use serde::{Deserialize, Serialize};

/// Type-erased wrapper for any Authenticator implementation
#[derive(Clone)]
pub struct SharedAuthenticator(pub Arc<dyn Authenticator>);

impl SharedAuthenticator {
    pub fn new<A: Authenticator + 'static>(authenticator: A) -> Self {
        Self(Arc::new(authenticator))
    }

    pub fn from_arc(authenticator: Arc<dyn Authenticator>) -> Self {
        Self(authenticator)
    }
}

#[async_trait::async_trait]
impl Authenticator for SharedAuthenticator {
    async fn authenticate(&self, context: &AuthContext) -> Result<AuthPrincipal, A2AError> {
        self.0.authenticate(context).await
    }

    fn security_scheme(&self) -> &SecurityScheme {
        self.0.security_scheme()
    }

    fn validate_context(&self, context: &AuthContext) -> Result<(), A2AError> {
        self.0.validate_context(context)
    }
}

/// Dynamic API Key authenticator wrapper for AXUM HTTP server
#[derive(Clone)]
pub struct ApiKeyAuthenticator {
    keys: Vec<String>,
    scheme: SecurityScheme,
}

impl ApiKeyAuthenticator {
    pub fn new(keys: Vec<String>, location: &str, name: &str) -> Self {
        Self {
            keys,
            scheme: SecurityScheme::ApiKey {
                name: name.to_string(),
                location: location.to_string(),
                description: Some("API Key Authentication".to_string()),
            },
        }
    }
}

#[async_trait::async_trait]
impl Authenticator for ApiKeyAuthenticator {
    async fn authenticate(&self, context: &AuthContext) -> Result<AuthPrincipal, A2AError> {
        self.validate_context(context)?;

        if self.keys.iter().any(|k| k == &context.credential) {
            let key_suffix = if context.credential.len() >= 4 {
                &context.credential[context.credential.len() - 4..]
            } else {
                &context.credential
            };
            Ok(AuthPrincipal::new(
                format!("apikey_principal_...{}", key_suffix),
                "apikey".to_string(),
            ))
        } else {
            Err(A2AError::Internal(
                "API key authentication failed: invalid API key".to_string(),
            ))
        }
    }

    fn security_scheme(&self) -> &SecurityScheme {
        &self.scheme
    }

    fn validate_context(&self, context: &AuthContext) -> Result<(), A2AError> {
        if context.scheme_type != "apikey"
            && context.scheme_type != "bearer"
            && context.scheme_type != "header"
        {
            return Err(A2AError::Internal(format!(
                "Invalid authentication scheme for API key: expected 'apikey', 'bearer', or 'header', got '{}'",
                context.scheme_type
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "builtin-jwt")]
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

#[cfg(feature = "builtin-jwt")]
#[derive(Clone)]
pub struct OAuth2JwtAuthenticator {
    secret: String,
    audience: String,
    issuer: String,
    scheme: SecurityScheme,
}

#[cfg(feature = "builtin-jwt")]
impl OAuth2JwtAuthenticator {
    pub fn try_new(
        secret: impl Into<String>,
        audience: impl Into<String>,
        issuer: impl Into<String>,
    ) -> Result<Self, anyhow::Error> {
        let secret = secret.into();
        let audience = audience.into();
        let issuer = issuer.into();

        if secret.is_empty() {
            anyhow::bail!("OAuth2JwtAuthenticator: secret cannot be empty");
        }
        if audience.is_empty() {
            anyhow::bail!("OAuth2JwtAuthenticator: audience cannot be empty (silent disable forbidden)");
        }
        if issuer.is_empty() {
            anyhow::bail!("OAuth2JwtAuthenticator: issuer cannot be empty");
        }

        Ok(Self {
            secret,
            audience,
            issuer,
            scheme: SecurityScheme::Http {
                scheme: "bearer".to_string(),
                bearer_format: Some("JWT".to_string()),
                description: Some("OAuth2 JWT Bearer Token".to_string()),
            },
        })
    }

    /// Legacy convenience constructor. Panics if arguments are invalid.
    pub fn new(secret: &str, audience: String, issuer: String) -> Self {
        Self::try_new(secret, audience, issuer)
            .expect("Invalid OAuth2JwtAuthenticator configuration")
    }
}

#[cfg(feature = "builtin-jwt")]
#[async_trait::async_trait]
impl Authenticator for OAuth2JwtAuthenticator {
    async fn authenticate(&self, context: &AuthContext) -> Result<AuthPrincipal, A2AError> {
        self.validate_context(context)?;

        let mut validation = jsonwebtoken::Validation::new(jsonwebtoken::Algorithm::HS256);
        validation.set_audience(&[&self.audience]);
        validation.set_issuer(&[&self.issuer]);
        validation.validate_exp = true;
        validation.validate_nbf = true;
        // Leeway set explicitly to 60 seconds
        validation.leeway = 60;

        let decoding_key = jsonwebtoken::DecodingKey::from_secret(self.secret.as_bytes());
        match jsonwebtoken::decode::<JwtClaims>(&context.credential, &decoding_key, &validation) {
            Ok(token_data) => {
                let principal_id = token_data
                    .claims
                    .tenant_id
                    .or(token_data.claims.sub)
                    .unwrap_or_else(|| "anonymous".to_string());
                Ok(AuthPrincipal::new(principal_id, "bearer".to_string()))
            }
            Err(e) => Err(A2AError::Internal(format!(
                "OAuth2 JWT verification failed: {}",
                e
            ))),
        }
    }

    fn security_scheme(&self) -> &SecurityScheme {
        &self.scheme
    }

    fn validate_context(&self, context: &AuthContext) -> Result<(), A2AError> {
        if context.scheme_type != "bearer" {
            return Err(A2AError::Internal(format!(
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
    /// No authentication (development/testing)
    None,
    /// Bearer token authentication
    BearerToken {
        tokens: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
    /// API Key authentication
    ApiKey {
        keys: Vec<String>,
        #[serde(default = "default_api_key_location")]
        location: String,
        #[serde(default = "default_api_key_name")]
        name: String,
    },
    /// OAuth2 JWT Bearer authentication
    #[cfg(feature = "builtin-jwt")]
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
    /// Create auth config from environment variables.
    /// Returns error if AUTH_JWT_SECRET is set without audience or issuer.
    pub fn try_from_env() -> Result<Self, anyhow::Error> {
        #[cfg(feature = "builtin-jwt")]
        {
            if let Ok(secret) = std::env::var("AUTH_JWT_SECRET") {
                let audience = std::env::var("AUTH_JWT_AUDIENCE")
                    .map_err(|_| anyhow::anyhow!("AUTH_JWT_SECRET set but AUTH_JWT_AUDIENCE is missing"))?;
                let issuer = std::env::var("AUTH_JWT_ISSUER")
                    .map_err(|_| anyhow::anyhow!("AUTH_JWT_SECRET set but AUTH_JWT_ISSUER is missing"))?;
                if audience.is_empty() || issuer.is_empty() {
                    anyhow::bail!("AUTH_JWT_AUDIENCE and AUTH_JWT_ISSUER cannot be empty");
                }
                return Ok(Self::OAuth2Jwt { secret, audience, issuer });
            }
        }

        if let Ok(tokens_str) = std::env::var("AUTH_BEARER_TOKENS") {
            let tokens: Vec<String> = tokens_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            if !tokens.is_empty() {
                return Ok(Self::BearerToken {
                    tokens,
                    format: std::env::var("AUTH_BEARER_FORMAT").ok(),
                });
            }
        }

        if let Ok(keys_str) = std::env::var("AUTH_API_KEYS") {
            let keys: Vec<String> = keys_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            if !keys.is_empty() {
                return Ok(Self::ApiKey {
                    keys,
                    location: std::env::var("AUTH_API_KEY_LOCATION")
                        .unwrap_or_else(|_| default_api_key_location()),
                    name: std::env::var("AUTH_API_KEY_NAME").unwrap_or_else(|_| default_api_key_name()),
                });
            }
        }

        Ok(Self::None)
    }

    /// Legacy from_env that falls back or logs errors
    pub fn from_env() -> Self {
        Self::try_from_env().unwrap_or(Self::None)
    }

    /// Convert AuthConfig into a SharedAuthenticator, if authentication is active
    pub fn to_shared_authenticator(&self) -> Result<Option<SharedAuthenticator>, anyhow::Error> {
        match self {
            AuthConfig::None => Ok(None),
            AuthConfig::BearerToken { tokens, .. } => {
                let auth = a2a_rs::adapter::BearerTokenAuthenticator::new(tokens.clone());
                Ok(Some(SharedAuthenticator::new(auth)))
            }
            AuthConfig::ApiKey { keys, location, name } => {
                let auth = ApiKeyAuthenticator::new(keys.clone(), location, name);
                Ok(Some(SharedAuthenticator::new(auth)))
            }
            #[cfg(feature = "builtin-jwt")]
            AuthConfig::OAuth2Jwt { secret, audience, issuer } => {
                let auth = OAuth2JwtAuthenticator::try_new(secret, audience, issuer)?;
                Ok(Some(SharedAuthenticator::new(auth)))
            }
        }
    }
}

fn default_api_key_location() -> String {
    "header".to_string()
}

fn default_api_key_name() -> String {
    "X-API-Key".to_string()
}
