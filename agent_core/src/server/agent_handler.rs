//! Simple agent handler for examples and testing
//!
//! This provides a complete agent implementation that bundles all business capabilities
//! (message handling, task management, notifications, and streaming) with in-memory storage.
//!
//! For production agents, you typically want to implement your own message handler
//! and compose it with the storage implementations directly.

// Example : https://github.com/EmilLindfors/a2a-rs/blob/master/http_client_server.rs

use std::sync::{Arc};
use tokio::sync::Mutex;

use async_trait::async_trait;

use a2a_rs::{
    ListTasksResult,
    adapter::storage::InMemoryTaskStorage,
    domain::{
        A2AError, Message, Part as MessagePart, Task, TaskArtifactUpdateEvent,
        TaskPushNotificationConfig, TaskState, TaskStatusUpdateEvent,
        ListTasksParams,
    },
    port::{
        AsyncMessageHandler, AsyncNotificationManager, AsyncStreamingHandler, AsyncTaskManager,
        streaming_handler::Subscriber,
    },
};

use crate::business_logic::agent::{Agent};
use agent_models::execution::execution_result::{ExecutionResult};
use crate::interaction_handler::InteractionHandler;
use crate::session::SessionStore;

#[derive(Clone)]
pub struct AgentHandler <T: Agent> {
    agent: Arc<Mutex<T>>,
    storage: Arc<InMemoryTaskStorage>,
    session_store: Arc<SessionStore>,
    interaction_handler: Arc<InteractionHandler>,
}

impl<T: Agent> AgentHandler<T> {
    pub fn new(agent:T) -> Self {
        let session_store = Arc::new(SessionStore::new());
        let interaction_handler = Arc::new(InteractionHandler::new(session_store.clone()));

        Self {
            agent: Arc::new(Mutex::new(agent)),
            storage: Arc::new(InMemoryTaskStorage::new()),
            session_store,
            interaction_handler,
        }
    }

    pub fn with_storage(
        agent:T,
        storage: InMemoryTaskStorage,
    ) -> Self {
        let session_store = Arc::new(SessionStore::new());
        let interaction_handler = Arc::new(InteractionHandler::new(session_store.clone()));
       
        Self {
            agent: Arc::new(Mutex::new(agent)),
            storage: Arc::new(storage),
            session_store,
            interaction_handler,
        }
    }

    #[allow(dead_code)]
    pub fn storage(&self) -> &Arc<InMemoryTaskStorage> {
        &self.storage
    }

    fn llm_message_to_a2a_message(&self, content: String) -> Result<Message, A2AError> {
        let message_id = uuid::Uuid::new_v4().to_string();
        let llm_msg = Message::agent_text(content, message_id);
        Ok(llm_msg)
    }
}

#[async_trait]
impl<T: Agent> AsyncMessageHandler for AgentHandler<T> {

    async fn process_message(
            &self,
            task_id: &str,
            message: &Message,
            session_id: Option<&str>,
        ) -> Result<Task, A2AError> {

        let session_id = session_id.unwrap_or("default_session").to_string();
        let _task = self.create_task(task_id, "context_task").await?;

        let user_query = message
            .parts
            .iter()
            .filter_map(|part| match part {
                MessagePart::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        let gemini_request = match self.interaction_handler.process_request(&session_id, user_query).await {
            Ok(req) => req,
            Err(e) => {
                tracing::error!("Interaction handler failed: {}", e);
                let error_msg = Message::agent_text(
                    format!("Interaction handler error: {}", e),
                    uuid::Uuid::new_v4().to_string(),
                );
                let task = self
                    .update_task_status(task_id, TaskState::Failed, Some(error_msg))
                    .await?;
                return Ok(task);
            }
        };

        let execution_result: ExecutionResult = match self.agent.lock().await.handle_request(gemini_request, message.metadata.clone()).await {
            Ok(result) => result,
            Err(e) => {
                tracing::error!("Agent execution failed: {}", e);
                let error_msg = Message::agent_text(
                    format!("Agent execution error: {}", e),
                    uuid::Uuid::new_v4().to_string(),
                );
                let task = self
                    .update_task_status(task_id, TaskState::Failed, Some(error_msg))
                    .await?;
                return Ok(task);
            }
        };

        let response_message = self.llm_message_to_a2a_message(execution_result.output.to_string())?;

        let task = self
            .update_task_status(task_id, TaskState::Completed, Some(response_message))
            .await?;
        
        Ok(task)
    }
}

#[async_trait]
impl<T: Agent> AsyncTaskManager for AgentHandler<T> {

    async fn create_task(
            &self, 
            task_id: &str, 
            context_id: &str
        ) -> Result<Task, A2AError> {

        self.storage.create_task(task_id, context_id).await
    }

    async fn get_task(
        &self,
        task_id:  &str,
        history_length: Option<u32>,
    ) -> Result<Task, A2AError> {
        self.storage.get_task(task_id, history_length).await
    }

    async fn update_task_status(
        &self,
        task_id: &str,
        state: TaskState,
        message: Option<Message>,
    ) -> Result<Task, A2AError> {
        self.storage
            .update_task_status(task_id, state, message)
            .await
    }

    async fn cancel_task(&self, task_id: & str) -> Result<Task, A2AError> {
        self.storage.cancel_task(task_id).await
    }

    async fn task_exists(&self, task_id: & str) -> Result<bool, A2AError> {
        self.storage.task_exists(task_id).await
    }

    async fn list_tasks_v3(
        &self,
        params: & ListTasksParams, 
    ) -> Result<ListTasksResult, A2AError> {
        self.storage.list_tasks_v3(params).await
    }
}

#[async_trait]
impl<T: Agent> AsyncNotificationManager for AgentHandler<T> {

    async fn set_task_notification(
        &self,
        config: & TaskPushNotificationConfig,
    ) -> Result<TaskPushNotificationConfig, A2AError> {
        self.storage.set_task_notification(config).await
    }

    async fn get_task_notification(
        &self,
        task_id: & str,
    ) -> Result<TaskPushNotificationConfig, A2AError> {
        self.storage.get_task_notification(task_id).await
    }

    async fn remove_task_notification(&self, task_id: & str) -> Result<(), A2AError> {
        self.storage.remove_task_notification(task_id).await
    }
}

#[async_trait]
impl<T: Agent> AsyncStreamingHandler for AgentHandler<T> {

    async fn add_status_subscriber(
        &self,
        task_id: & str,
        subscriber: Box<dyn Subscriber<TaskStatusUpdateEvent> + Send + Sync>,
    ) -> Result<String, A2AError> {
        self.storage
            .add_status_subscriber(task_id, subscriber)
            .await
    }

    async fn add_artifact_subscriber(
        &self,
        task_id: & str,
        subscriber: Box<dyn Subscriber<TaskArtifactUpdateEvent> + Send + Sync>,
    ) -> Result<String, A2AError> {
        self.storage
            .add_artifact_subscriber(task_id, subscriber)
            .await
    }

    async fn remove_subscription(&self, subscription_id: & str) -> Result<(), A2AError> {
        self.storage.remove_subscription(subscription_id).await
    }

    async fn remove_task_subscribers(&self, task_id: & str) -> Result<(), A2AError> {
        self.storage.remove_task_subscribers(task_id).await
    }

    async fn get_subscriber_count(&self, task_id: & str) -> Result<usize, A2AError> {
        self.storage.get_subscriber_count(task_id).await
    }

    async fn broadcast_status_update(
        &self,
        task_id: & str,
        update: TaskStatusUpdateEvent,
    ) -> Result<(), A2AError> {
        self.storage.broadcast_status_update(task_id, update).await
    }

    async fn broadcast_artifact_update(
        &self,
        task_id: & str,
        update: TaskArtifactUpdateEvent,
    ) -> Result<(), A2AError> {
        self.storage.broadcast_artifact_update(task_id, update).await
    }

    async fn status_update_stream(
        &self,
        task_id: & str,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<TaskStatusUpdateEvent, A2AError>> + Send>,
        >,
        A2AError,
    > {
        self.storage.status_update_stream(task_id).await
    }

    async fn artifact_update_stream(
        &self,
        task_id: & str,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<TaskArtifactUpdateEvent, A2AError>> + Send>,
        >,
        A2AError,
    > {
        self.storage.artifact_update_stream(task_id).await
    }

    async fn combined_update_stream(
        &self,
        task_id: & str,
    ) -> Result<
        std::pin::Pin<
            Box<
                dyn futures::Stream<
                        Item = Result<a2a_rs::port::streaming_handler::UpdateEvent, A2AError>,
                    > + Send,
            >,
        >,
        A2AError,
    > {
        self.storage.combined_update_stream(task_id).await
    }
}
