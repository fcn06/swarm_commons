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
    *   **Purpose:** Contains the fundamental traits, structures, and business logic that define what an "agent" is within the Swarm framework. This includes core agent behavior, interaction patterns, and the basic mechanisms for processing requests and generating responses.
    *   **Key Features:** Defines the `Agent` trait, common agent lifecycle methods, and core request/response models.
*   **`agent_models`**:
    *   **Purpose:** Houses the shared data models and domain-specific structures used for communication and state management across different agents and services. This includes models for evaluation, execution results, factory configurations, graph definitions for workflows, and memory structures.
    *   **Key Features:** `EvaluationModels`, `ExecutionResult`, `FactoryConfig`, `GraphDefinition`, `HighLevelPlanDefinition`, `MemoryModels`, `RegistryModels`.
*   **`configuration`**:
    *   **Purpose:** Manages the various configuration structures, prompt templates, and settings required by different components of the Swarm framework. This ensures that agents and services can be easily configured and customized.
    *   **Key Features:** Contains TOML configuration files (e.g., `agent_basic_config.toml`, `factory_config.toml`), and prompt templates (e.g., `detailed_workflow_agent_prompt.txt`).
*   **`llm_api`**:
    *   **Purpose:** Provides a standardized interface and client implementations for interacting with various Large Language Models (LLMs). This abstracts away the specifics of different LLM providers, allowing agents to use LLMs consistently.
    *   **Key Features:** Defines common LLM client traits, handles API key management, and facilitates chat and tool-calling interactions with LLMs (e.g., Groq, Google AI Studio Gemini).

## **Usage**

To use any of the crates within `swarm_commons`, add them as dependencies in your `Cargo.toml`:

```toml
[dependencies]
swarm_commons_agent_core = { path = "../swarm_commons/agent_core" }
swarm_commons_agent_models = { path = "../swarm_commons/agent_models" }
swarm_commons_configuration = { path = "../swarm_commons/configuration" }
swarm_commons_llm_api = { path = "../swarm_commons/llm_api" }
```

(Note: You might need to adjust the path based on your project structure if you're using these as local path dependencies.)

## **Contributing**

We welcome contributions to `swarm_commons`! By contributing to these foundational crates, you help strengthen the entire Swarm ecosystem. Please refer to the main Swarm project's contribution guidelines for more details.
