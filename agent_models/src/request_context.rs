use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Provider-agnostic parsed request context extracted from metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RequestContext {
    pub tenant_id: Option<String>,
    pub thread_id: Option<String>,
    pub credential: Option<String>,
    pub extra: Map<String, Value>,
}

impl RequestContext {
    pub fn from_metadata(m: Option<&Map<String, Value>>) -> Self {
        let mut ctx = Self::default();
        let Some(metadata) = m else {
            return ctx;
        };

        // Extract tenant_id from "tenant_id" or "tenant"
        if let Some(t) = metadata.get("tenant_id").or_else(|| metadata.get("tenant")) {
            if let Some(s) = t.as_str() {
                ctx.tenant_id = Some(s.to_string());
            }
        }

        // Extract thread_id from "thread_id" or "conversation_id"
        if let Some(t) = metadata.get("thread_id").or_else(|| metadata.get("conversation_id")) {
            if let Some(s) = t.as_str() {
                ctx.thread_id = Some(s.to_string());
            }
        }

        // Extract credential from "agent_jwt", "session_jwt", or "authorization"
        if let Some(c) = metadata
            .get("agent_jwt")
            .or_else(|| metadata.get("session_jwt"))
            .or_else(|| metadata.get("authorization"))
        {
            if let Some(s) = c.as_str() {
                let s_trimmed = s.trim();
                let token = if let Some(stripped) = s_trimmed.strip_prefix("Bearer ") {
                    stripped.trim().to_string()
                } else if let Some(stripped) = s_trimmed.strip_prefix("bearer ") {
                    stripped.trim().to_string()
                } else {
                    s_trimmed.to_string()
                };
                ctx.credential = Some(token);
            }
        }

        // Keep all other/extra entries
        for (k, v) in metadata {
            if k != "tenant_id"
                && k != "tenant"
                && k != "thread_id"
                && k != "conversation_id"
                && k != "agent_jwt"
                && k != "session_jwt"
                && k != "authorization"
            {
                ctx.extra.insert(k.clone(), v.clone());
            }
        }

        ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_metadata_aliases() {
        let mut map = Map::new();
        map.insert("tenant".into(), Value::String("tenant_123".into()));
        map.insert("conversation_id".into(), Value::String("conv_456".into()));
        map.insert(
            "authorization".into(),
            Value::String("Bearer secret_jwt".into()),
        );
        map.insert("custom_key".into(), Value::Bool(true));

        let ctx = RequestContext::from_metadata(Some(&map));
        assert_eq!(ctx.tenant_id.as_deref(), Some("tenant_123"));
        assert_eq!(ctx.thread_id.as_deref(), Some("conv_456"));
        assert_eq!(ctx.credential.as_deref(), Some("secret_jwt"));
        assert_eq!(ctx.extra.get("custom_key"), Some(&Value::Bool(true)));
    }
}
