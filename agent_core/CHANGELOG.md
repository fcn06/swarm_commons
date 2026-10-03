# Changelog

All notable changes to `agent_core` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-03

### Added
- **`SessionKeyResolver` & Multi-Tenant Session Isolation (WP1)**:
  - Added `SessionKeyResolver` trait for pluggable session key resolution strategies.
  - Added `A2aSessionKeyResolver` which isolates turns and returns `None` for requests lacking an explicit session identifier, eliminating cross-tenant leakage via `default_session` (F2).
  - Added `TenantThreadResolver` implementing `{tenant_id}_{thread_id}` composite keys with `_` separation.
  - Added `SessionStore::with_limits(max_items, ttl)` supporting bounded session history, FIFO trimming, and background TTL-based eviction (F4).
  - Added `AgentHandlerBuilder` (`AgentHandler::builder(agent)`) to support customizable session resolvers and stores.
- **Pluggable & Secure Authentication (WP2)**:
  - Added `SharedAuthenticator(pub Arc<dyn Authenticator>)` wrapper implementing `a2a_rs::Authenticator` for custom or third-party authenticator injection without forks.
  - Added `SecureAgentServerBuilder` (`SecureAgentServer::builder(agent, config)`) supporting `.with_authenticator(...)`, `.with_storage(...)`, `.with_discovery(...)`, `.with_nats(...)`, and graceful shutdown signals.
  - Hardened `OAuth2JwtAuthenticator`: enforced algorithm pinning to HS256, mandatory audience and issuer validation, expiration (`exp`) and not-before (`nbf`) checks with 60-second leeway, and safe constructor `try_new`.
  - Added public discovery helpers `register_with_discovery_service`, `build_agent_info`, and `build_agent_definition`.
  - Added `parse_bind_address` using `url::Url` for robust endpoint parsing across `http://`, `https://`, and raw `host:port` formats (F10).
- **Native In-Process NATS Transport (WP3)**:
  - Added `NatsTransport` and `NatsDispatchConfig` under the `nats` crate feature.
  - Direct in-process turn execution via `DefaultRequestProcessor::process_raw_request` without HTTP loopback.
  - Full authentication parity via `CredentialExtractor` prior to in-process dispatch.
- **Gateway Hardening & Tool Forwarding (WP4)**:
  - Added `GatewayTurnOptions` to `GatewayBackend::process_turn_with_options` and `process_turn_stream_with_options` to pass tools and tool choice configurations.
  - Inbound authentication support on `GatewayServer` via optional `SharedAuthenticator` enforcing bearer token validation on `/v1/responses` and `/v1/chat/completions` (F7).
  - Multi-provider fallback retry loop in `MultiModelGatewayBackend`.
  - Structured HTTP error mappings: circuit breaker open returns `503 Service Unavailable`, upstream rate limits map to `429 Too Many Requests`, gateway timeouts to `504`, upstream bad gateway to `502` (F8).
- **RequestContext (WP5)**:
  - Integration with `agent_models::RequestContext` for unified extraction of `tenant_id`, `thread_id`, and credentials across header and metadata aliases.
- **Safety**:
  - Added `#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]` across `agent_core` to guarantee zero `.unwrap()` or `.expect()` calls in production code paths.

### Changed
- **Removed Concurrency Bottleneck (F1)**:
  - Removed `Arc<Mutex<T>>` serialization in `AgentHandler`, unlocking concurrent turn processing across separate sessions and requests.
- **Assistant Response Persistence (F3)**:
  - `AgentHandler` and `interaction_handler` now automatically append generated assistant response items to session history upon turn completion.
- **Stateless Chat Completions (F5)**:
  - Removed unintended session mutations from `/v1/chat/completions`, aligning with the stateless OpenAI chat completions specification.
- **Tool Forwarding Preservation (F6)**:
  - Inbound tools and tool choices in `/v1/chat/completions` and `/v1/responses` are forwarded to upstream LLM providers (Google Gemini, Groq, OpenAI), and generated tool calls are preserved in both streaming and non-streaming responses.
- **Gateway Default Bind Address (F9)**:
  - Default bind address updated from `0.0.0.0:8080` to secure localhost `127.0.0.1:8080`.
