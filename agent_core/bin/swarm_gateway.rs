use std::sync::Arc;
use clap::Parser;
use tracing::info;

use agent_core::server::gateway_server::{
    GatewayBackend, GatewayConfigFile, GatewayResilienceSection, GatewayServer, MultiModelGatewayBackend,
};
use agent_core::session::SessionStore;
use configuration::setup_logging;

#[derive(Parser, Debug)]
#[clap(
    name = "swarm_gateway",
    author,
    version,
    about = "Swarm Standalone Ultra-Lean Model Gateway (OpenAI & Open Responses)"
)]
struct Args {
    /// Path to gateway configuration file (TOML format)
    #[clap(long, short = 'c')]
    config_file: Option<String>,

    /// Bind address (e.g. 127.0.0.1:8080)
    #[clap(long, default_value = "127.0.0.1:8080")]
    bind_address: String,

    /// Log level (trace, debug, info, warn, error)
    #[clap(long, default_value = "info")]
    log_level: String,

    /// Optional max concurrent in-flight requests (load shedding)
    #[clap(long)]
    max_concurrency: Option<usize>,

    /// Optional per-request execution timeout in seconds
    #[clap(long)]
    request_timeout_secs: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut bind_address = args.bind_address.clone();
    let mut log_level = args.log_level.clone();

    let (backend, resilience_config): (Arc<dyn GatewayBackend>, GatewayResilienceSection) = if let Some(config_path) = &args.config_file {
        match std::fs::read_to_string(config_path) {
            Ok(content) => match toml::from_str::<GatewayConfigFile>(&content) {
                Ok(config) => {
                    if let Some(server) = &config.server {
                        if let Some(addr) = &server.bind_address {
                            if args.bind_address == "127.0.0.1:8080" {
                                bind_address = addr.clone();
                            }
                        }
                        if let Some(level) = &server.log_level {
                            if args.log_level == "info" {
                                log_level = level.clone();
                            }
                        }
                    }
                    let mut res = config.resilience.clone().unwrap_or_default();
                    if let Some(mc) = args.max_concurrency {
                        res.max_concurrent_requests = Some(mc);
                    }
                    if let Some(to) = args.request_timeout_secs {
                        res.request_timeout_seconds = Some(to);
                    }
                    println!("✔ Loaded Gateway Configuration from: {}", config_path);
                    (Arc::new(MultiModelGatewayBackend::from_config(&config)), res)
                }
                Err(err) => {
                    eprintln!("⚠️ Failed to parse config file {}: {}. Using env defaults.", config_path, err);
                    let mut res = GatewayResilienceSection::default();
                    res.max_concurrent_requests = args.max_concurrency;
                    res.request_timeout_seconds = args.request_timeout_secs;
                    (Arc::new(MultiModelGatewayBackend::from_env()), res)
                }
            },
            Err(err) => {
                eprintln!("⚠️ Failed to read config file {}: {}. Using env defaults.", config_path, err);
                let mut res = GatewayResilienceSection::default();
                res.max_concurrent_requests = args.max_concurrency;
                res.request_timeout_seconds = args.request_timeout_secs;
                (Arc::new(MultiModelGatewayBackend::from_env()), res)
            }
        }
    } else {
        let mut res = GatewayResilienceSection::default();
        res.max_concurrent_requests = args.max_concurrency;
        res.request_timeout_seconds = args.request_timeout_secs;
        (Arc::new(MultiModelGatewayBackend::from_env()), res)
    };

    setup_logging(&log_level);
    info!("Starting Swarm Standalone Gateway on {}", bind_address);

    let session_store = Arc::new(SessionStore::new());
    let gateway_server = GatewayServer::new(session_store, backend)
        .with_resilience(resilience_config.clone());

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║       🌐 fcn06/swarm Standalone Gateway Server (Mode 2)        ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ • Open Responses Route:      POST http://{}/v1/responses      ║", bind_address);
    println!("║ • Chat Completions Route:    POST http://{}/v1/chat/completions║", bind_address);
    println!("║ • Health Check Route:        GET  http://{}/health            ║", bind_address);
    println!("║ • In-Memory Session Storage: Active (DashMap lock-free)        ║");
    if let Some(limit) = resilience_config.max_concurrent_requests {
        println!("║ • Concurrency Limit:         {:<33} ║", format!("{} max inflight", limit));
    }
    if let Some(timeout) = resilience_config.request_timeout_seconds {
        println!("║ • Request Timeout:           {:<33} ║", format!("{} seconds", timeout));
    }
    if let Some(cfg) = &args.config_file {
        println!("║ • Config File:               {:<33} ║", cfg);
    }
    println!("╚════════════════════════════════════════════════════════════════╝");

    gateway_server.start(&bind_address).await?;

    Ok(())
}
