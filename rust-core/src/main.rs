//! localai-core — Ollama proxy + SHA-256 response cache
//!
//! Routes:
//!   POST   /generate   — proxy to Ollama, cache response
//!   DELETE /cache      — clear in-memory cache
//!
//! Env vars:
//!   OLLAMA_URL   (default: http://localhost:11434)
//!   PORT         (default: 8080)

use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, post},
    Json, Router,
};
use bytes::Bytes;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Cache = Arc<Mutex<HashMap<String, String>>>;

#[derive(Deserialize)]
struct GenerateReq {
    model: String,
    prompt: String,
    /// If true, stream raw chunks back to caller instead of waiting for full response.
    #[serde(default)]
    stream: bool,
}

#[derive(Serialize)]
struct GenerateResp {
    response: String,
    cached: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cache_key(model: &str, prompt: &str) -> String {
    let mut h = Sha256::new();
    h.update(model.as_bytes());
    h.update(b"|");
    h.update(prompt.as_bytes());
    format!("{:x}", h.finalize())
}

fn ollama_url() -> String {
    std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Non-streaming: wait for full Ollama response, cache it, return JSON.
async fn generate_buffered(
    cache: &Cache,
    req: &GenerateReq,
) -> Result<Json<GenerateResp>, (StatusCode, String)> {
    let key = cache_key(&req.model, &req.prompt);

    // Cache hit — return immediately
    if let Some(hit) = cache.lock().unwrap().get(&key).cloned() {
        return Ok(Json(GenerateResp { response: hit, cached: true }));
    }

    let body = serde_json::json!({
        "model":  req.model,
        "prompt": req.prompt,
        "stream": false,
    });

    let resp = reqwest::Client::new()
        .post(format!("{}/api/generate", ollama_url()))
        .json(&body)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    let val: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    let text = val["response"].as_str().unwrap_or("").to_string();

    cache.lock().unwrap().insert(key, text.clone());

    Ok(Json(GenerateResp { response: text, cached: false }))
}

/// Streaming: forward Ollama's NDJSON chunks directly to the HTTP response body.
/// Cache is NOT used for streamed requests (no complete response to store).
async fn generate_streamed(
    req: &GenerateReq,
) -> Result<Response, (StatusCode, String)> {
    let body = serde_json::json!({
        "model":  req.model,
        "prompt": req.prompt,
        "stream": true,
    });

    let upstream = reqwest::Client::new()
        .post(format!("{}/api/generate", ollama_url()))
        .json(&body)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    // Map each chunk: parse the "response" token field, forward raw text bytes
    let stream = upstream.bytes_stream().map(|chunk| {
        let chunk = chunk.map_err(|e| e.to_string())?;
        if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&chunk) {
            let token = val["response"].as_str().unwrap_or("").to_string();
            Ok::<Bytes, String>(Bytes::from(token))
        } else {
            Ok(Bytes::new())
        }
    });

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/plain; charset=utf-8")
        .body(Body::from_stream(stream))
        .unwrap())
}

async fn handle_generate(
    State(cache): State<Cache>,
    Json(req): Json<GenerateReq>,
) -> Result<Response, (StatusCode, String)> {
    if req.stream {
        generate_streamed(&req).await
    } else {
        generate_buffered(&cache, &req).await.map(|j| j.into_response())
    }
}

async fn handle_clear_cache(State(cache): State<Cache>) -> impl IntoResponse {
    let n = {
        let mut c = cache.lock().unwrap();
        let n = c.len();
        c.clear();
        n
    };
    format!("cleared {n} entries")
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cache: Cache = Arc::new(Mutex::new(HashMap::new()));

    let app = Router::new()
        .route("/generate", post(handle_generate))
        .route("/cache",    delete(handle_clear_cache))
        .with_state(cache);

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".into());
    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr).await.unwrap();
    println!("localai-core on {addr}  (ollama → {})", ollama_url());
    axum::serve(listener, app).await.unwrap();
}
