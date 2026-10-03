# 🦀 Swarm Commons: Foundational Crates for Swarm Agents 🦀

> **Swarm Commons** provides the essential, shared building blocks and fundamental abstractions for developing intelligent agents within the Swarm ecosystem. This crate consolidates core functionalities, communication protocols, and configuration structures that are common across various agent types and services, promoting consistency, reusability, and efficient development.

## **Why Swarm Commons?**

As the Swarm framework evolves, certain foundational components became universally necessary for any agent or service interacting within the system. By centralizing these into `swarm_commons`, we aim to:

*   **Promote Reusability:** Avoid duplicating core logic across different agent implementations.
*   **Enhance Consistency:** Ensure all agents and services adhere to the same underlying models and protocols.
*   **Simplify Development:** Provide a stable and well-defined set of common utilities, allowing developers to focus on domain-specific agent intelligence.
*   **Facilitate Evolution:** Decouple core abstractions from specific agent implementations, making it easier to evolve the Swarm framework.

## **Included Crates & Their Purpose**

`swarm_commons` is a workspace containing several foundational crates:

*   **`agent_core`**:
    *   **Purpose:** Contains the fundamental traits, structures, and business logic that define what an "agent" is within the Swarm framework. This includes core agent behavior, interaction patterns, session management, secure transport, and the basic mechanisms for processing requests and generating responses.
    *   **Key Features (v0.2.0):**
        *   **Turn Concurrency:** Fully concurrent, lock-free turn execution across sessions (removed `Mutex` serialization bottleneck in `AgentHandler`).
        *   **Session Isolation & Key Resolution:** Pluggable `SessionKeyResolver` trait, `A2aSessionKeyResolver` (eliminates cross-tenant contamination by rejecting requests lacking explicit session IDs), and `TenantThreadResolver` (`{tenant}_{thread}` composite keys).
        *   **Session Storage Limits & TTL:** Bounded FIFO session history, configurable capacity limits, and automatic background TTL-based eviction (`SessionStore::with_limits(max_items, ttl)`).
        *   **Assistant Response History:** Automatic persistence of assistant turn responses into session history upon completion.
        *   **Pluggable & Hardened Authentication:** `SharedAuthenticator` (`Arc<dyn Authenticator>`) for custom auth integration; hardened `OAuth2JwtAuthenticator` with pinned HS256 algorithm, mandatory audience/issuer enforcement, and 60-second clock skew leeway; robust `parse_bind_address` supporting `http://`, `https://`, and raw `host:port`.
        *   **Native In-Process NATS Transport:** Optional `nats` feature with zero HTTP loopback, direct `DefaultRequestProcessor::process_raw_request` dispatch, and subject/header credential validation via `CredentialExtractor`.
        *   **Hardened Gateway Server (`swarm_gateway`):**
            *   Stateless `/v1/chat/completions` (OpenAI specification compliance).
            *   Stateful `/v1/responses` with session isolation.
            *   Full tool forwarding and preservation (`GatewayTurnOptions`) for tools and tool choices across streaming SSE and non-streaming responses.
            *   Inbound bearer auth middleware (`with_authenticator`) protecting API endpoints while keeping `/health` public.
            *   Multi-provider fallback retry loops and structured HTTP status mappings (503 for open circuit breakers, 429 for rate limits, 502 for upstream errors, 504 for timeouts).
            *   Secure default bind address: `127.0.0.1:8080`.
        *   **Ergonomic Builders:** `AgentHandlerBuilder` and `SecureAgentServerBuilder`.
        *   **Production Safety:** Enforced `#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]` for zero-panic code paths.
*   **`agent_models`**:
    *   **Purpose:** Houses the shared data models and domain-specific structures used for communication and state management across different agents and services.
    *   **Key Features:**
        *   `RequestContext`: Unified extraction and normalization of `tenant_id` (`tenant_id`/`tenant`), `thread_id` (`thread_id`/`conversation_id`), and authorization tokens (`agent_jwt`/`session_jwt`/`authorization`) across headers and metadata.
        *   `AgentRequest::context(&self)`: Fluent accessor for normalized `RequestContext`.
        *   Evaluation, Execution, Factory, Graph, Plan, Memory, and Registry models (`EvaluationModels`, `ExecutionResult`, `FactoryConfig`, `GraphDefinition`, `HighLevelPlanDefinition`, `MemoryModels`, `RegistryModels`).
*   **`configuration`**:
    *   **Purpose:** Manages configuration structures, prompt templates, and settings required by different components of the Swarm framework.
    *   **Key Features:** Contains TOML configuration files (e.g., `agent_basic_config.toml`, `factory_config.toml`, `gateway_config.toml`), and prompt templates (e.g., `detailed_workflow_agent_prompt.txt`).
*   **`llm_api`**:
    *   **Purpose:** Provides a standardized interface and client implementations for interacting with various Large Language Models (LLMs), abstracting provider specifics.
    *   **Key Features:** Defines common LLM client traits, API key management, tool calling, and chat interactions with cloud and local backends (Groq, Google Gemini, OpenAI, Ollama, vLLM).

## **Feature Flags**

The `agent_core` crate provides configurable feature flags:

*   `default = ["builtin-jwt", "gateway"]` — Standard server and gateway capabilities with built-in JWT verification.
*   `builtin-jwt` — Enables JWT-based authentication via `jsonwebtoken`.
*   `gateway` — Compiles the standalone `swarm_gateway` binary and gateway routing module.
*   `nats` — Enables native in-process NATS transport (`async-nats`) without HTTP loopbacks.

## **Usage**

To use any of the crates within `swarm_commons`, add them as dependencies in your `Cargo.toml`:

```toml
[dependencies]
agent_core = { path = "../swarm_commons/agent_core", features = ["gateway", "nats"] }
agent_models = { path = "../swarm_commons/agent_models" }
configuration = { path = "../swarm_commons/configuration" }
llm_api = { path = "../swarm_commons/llm_api" }
```

(Note: Adjust paths based on your workspace or local project structure.)

## **Changelog**

For a detailed history of changes, see [agent_core/CHANGELOG.md](agent_core/CHANGELOG.md).

## **Contributing**

We welcome contributions to `swarm_commons`! By contributing to these foundational crates, you help strengthen the entire Swarm ecosystem. Please refer to the main Swarm project's contribution guidelines for more details.
