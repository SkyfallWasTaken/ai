use actix_web::{
    get,
    http::StatusCode,
    middleware::Logger,
    post,
    web::{self, Bytes},
    App, HttpRequest, HttpResponse, HttpServer, Responder,
};
use async_stream::stream;
use deadpool_postgres::{Config, ManagerConfig, Pool, RecyclingMethod, Runtime};
use futures::StreamExt;
use minijinja::{context, path_loader, Environment};
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ModelObject {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelObject>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Function {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: Function,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
enum ToolChoice {
    String(String),
    Object {
        #[serde(rename = "type")]
        tool_type: String,
        function: HashMap<String, String>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_reasoning: Option<bool>,
}

#[derive(Clone)]
struct AppState {
    client: Client,
    db_pool: Option<Pool>,
}

use std::sync::Arc;

static AVAILABLE_MODELS: OnceLock<Vec<String>> = OnceLock::new();
static TEMPLATE_ENV: OnceLock<Arc<Environment>> = OnceLock::new();

fn get_available_models() -> &'static Vec<String> {
    AVAILABLE_MODELS.get_or_init(|| {
        std::env::var("COMPLETION_MODELS")
            .or_else(|_| std::env::var("COMPLETIONS_MODEL"))
            .unwrap_or_else(|_| "unknown".to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    })
}

fn get_template_env() -> &'static Arc<Environment<'static>> {
    TEMPLATE_ENV.get_or_init(|| {
        let mut env = Environment::new();
        env.set_loader(path_loader("templates"));
        Arc::new(env)
    })
}

fn log_request(
    pool: Pool,
    req: ChatRequest,
    res: Value,
    ip: std::net::IpAddr,
    user_agent: Option<String>,
) {
    tokio::spawn(async move {
        if let Ok(conn) = pool.get().await {
            let req_json = serde_json::to_string(&req).unwrap_or_default();
            let res_json = res.to_string();
            let _ = conn.execute(
                "INSERT INTO api_request_logs (request, response, ip, user_agent) VALUES ($1, $2, $3, $4)",
                &[&req_json, &res_json, &ip, &user_agent]
            ).await;
        }
    });
}

#[get("/")]
async fn index(data: web::Data<AppState>) -> Result<impl Responder, Box<dyn std::error::Error>> {
    let sum: i32 = if let Some(pool) = &data.db_pool {
        match pool.get().await {
            Ok(conn) => {
                match conn.query_one("SELECT COALESCE(SUM((response->'usage'->>'total_tokens')::INTEGER), 0) as total FROM api_request_logs WHERE response ? 'usage' AND response->'usage' ? 'total_tokens';", &[]).await {
                    Ok(row) => row.get::<_, i32>("total"),
                    Err(_) => -1
                }
            }
            Err(_) => -1,
        }
    } else {
        -1
    };

    let env = get_template_env();
    let tmpl = env.get_template("index.jinja")?;
    let available_models = get_available_models();
    let ctx = context!(total_tokens => sum, models => available_models);
    let page = tmpl.render(ctx)?;

    Ok(HttpResponse::Ok().content_type("text/html").body(page))
}

#[post("/chat/completions")]
async fn completions(
    data: web::Data<AppState>,
    mut body: web::Json<ChatRequest>,
    req: HttpRequest,
) -> Result<impl Responder, Box<dyn std::error::Error>> {
    let available_models = get_available_models();
    body.model = Some(
        body.model
            .as_ref()
            .filter(|m| available_models.contains(m))
            .cloned()
            .or_else(|| available_models.first().cloned())
            .unwrap_or_else(|| "unknown".to_string()),
    );

    let is_stream = body.stream == Some(true);
    let log_body = body.clone();
    let ip = req
        .peer_addr()
        .map(|a| a.ip())
        .unwrap_or_else(|| "0.0.0.0".parse().unwrap());
    let user_agent = req
        .headers()
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(String::from);

    let res = data
        .client
        .post(std::env::var("COMPLETIONS_URL")?)
        .json(&body.into_inner())
        .send()
        .await?;

    if !res.status().is_success() {
        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        return Ok(HttpResponse::build(status)
            .content_type("application/json")
            .body(res.text().await?));
    }

    if is_stream {
        let mut stream_res = res.bytes_stream();
        let pool = data.db_pool.clone();

        let processed_stream = stream! {
            let mut buffer = Vec::with_capacity(8192);
            while let Some(chunk) = stream_res.next().await {
                if let Ok(bytes) = chunk {
                    buffer.extend_from_slice(&bytes);
                    
                    while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                        let line_bytes: Vec<u8> = buffer.drain(..=pos).collect();
                        let line = String::from_utf8_lossy(&line_bytes[..line_bytes.len()-1]);
                        
                        if line.is_empty() { continue; }
                        let json_str = line.strip_prefix("data: ").unwrap_or(&line);
                        if json_str == "[DONE]" { break; }

                        if let Ok(val) = serde_json::from_str::<Value>(json_str) {
                            if let Some(ref pool) = pool {
                                log_request(pool.clone(), log_body.clone(), val.clone(), ip, user_agent.clone());
                            }
                            let mut output = Vec::with_capacity(json_str.len() + 10);
                            output.extend_from_slice(b"data: ");
                            if let Ok(response_bytes) = serde_json::to_vec(&val) {
                                output.extend_from_slice(&response_bytes);
                                output.extend_from_slice(b"\n\n");
                                yield Ok::<Bytes, Box<dyn std::error::Error>>(Bytes::from(output));
                            }
                        }
                    }
                }
            }
        };

        Ok(HttpResponse::Ok()
            .content_type("application/x-ndjson")
            .streaming(Box::pin(processed_stream)))
    } else {
        let res_json = res.json::<Value>().await?;
        if let Some(ref pool) = data.db_pool {
            log_request(pool.clone(), log_body, res_json.clone(), ip, user_agent);
        }
        Ok(HttpResponse::Ok()
            .content_type("application/json")
            .json(res_json))
    }
}

#[get("/models")]
async fn get_models() -> Result<impl Responder, Box<dyn std::error::Error>> {
    let available_models = get_available_models()
        .iter()
        .map(|id| ModelObject {
            id: id.clone(),
            object: "model".to_string(),
            created: 1640995200,
            owned_by: "system".to_string(),
        })
        .collect();

    Ok(HttpResponse::Ok().json(ModelsResponse {
        object: "list".to_string(),
        data: available_models,
    }))
}

#[get("/model")]
async fn get_model() -> impl Responder {
    HttpResponse::Ok().body(get_available_models().join(","))
}

#[get("/echo")]
async fn echo(req_body: String) -> impl Responder {
    HttpResponse::Ok().body(req_body)
}

#[get("/hey")]
async fn hey() -> impl Responder {
    HttpResponse::Ok().body("Hey there!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let db_pool = if let Ok(url) = std::env::var("DB_URL") {
        let mut cfg = Config::new();
        cfg.url = Some(url);
        cfg.manager = Some(ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        });
        cfg.pool = Some(deadpool_postgres::PoolConfig::new(32));

        match cfg.create_pool(Some(Runtime::Tokio1), tokio_postgres::NoTls) {
            Ok(pool) => {
                log::info!("Database pool created. Checking for table...");
                match pool.get().await {
                    Ok(conn) => {
                        let creation_result = conn
                            .batch_execute(
                                "CREATE TABLE IF NOT EXISTS api_request_logs (
                                id SERIAL PRIMARY KEY,
                                request JSONB NOT NULL,
                                response JSONB NOT NULL,
                                ip INET NOT NULL,
                                user_agent VARCHAR(512),
                                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                            );
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_logs_created_at ON api_request_logs(created_at);
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_logs_ip ON api_request_logs(ip);
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_logs_response_usage ON api_request_logs USING GIN ((response->'usage'));
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_logs_total_tokens ON api_request_logs(((response->'usage'->>'total_tokens')::INTEGER)) WHERE response ? 'usage' AND response->'usage' ? 'total_tokens';
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_logs_request_model ON api_request_logs USING GIN ((request->'model'));",
                            )
                            .await;

                        match creation_result {
                            Ok(_) => {
                                log::info!("Database table 'api_request_logs' is ready.");
                                Some(pool)
                            }
                            Err(e) => {
                                log::error!("Failed to create database table: {e}");
                                None
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to get connection from pool: {e}");
                        None
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to create database pool: {e}");
                None
            }
        }
    } else {
        log::warn!("DB_URL not set. Database logging is disabled.");
        None
    };

    let api_key = std::env::var("KEY").expect("KEY environment variable must be set");
    let mut headers = header::HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    headers.insert(header::AUTHORIZATION, {
        let mut token = header::HeaderValue::from_str(&format!("Bearer {api_key}")).unwrap();
        token.set_sensitive(true);
        token
    });

    let client = Client::builder()
        .default_headers(headers)
        .pool_max_idle_per_host(20)
        .pool_idle_timeout(std::time::Duration::from_secs(60))
        .timeout(std::time::Duration::from_secs(120))
        .connect_timeout(std::time::Duration::from_secs(10))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        .use_rustls_tls()
        .build()
        .unwrap();
    let app_state = AppState { client, db_pool };

    HttpServer::new(move || {
        App::new()
            .wrap(
                actix_cors::Cors::default()
                    .allow_any_origin()
                    .allow_any_method()
                    .allow_any_header()
                    .max_age(3600),
            )
            .app_data(web::Data::new(app_state.clone()))
            .app_data(web::JsonConfig::default().limit(1024 * 1024 * 10))
            .service(index)
            .service(completions)
            .service(get_models)
            .service(get_model)
            .service(echo)
            .service(hey)
            .wrap(Logger::default())
    })
    .workers(num_cpus::get())
    .client_request_timeout(std::time::Duration::from_secs(60))
    .client_disconnect_timeout(std::time::Duration::from_secs(5))
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
