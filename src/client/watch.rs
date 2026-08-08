//! Live status over WebSocket.
//!
//! A gateway broadcasts cluster-wide status on `/ws/cluster-status`; a worker broadcasts
//! its own on `/ws/status`. Which one to open is decided by [`WatchMode`], and `Auto` asks
//! `/health` first.

use futures_util::StreamExt;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::protocol::Message;
use url::Url;

use crate::client::{Client, stream::ChunkStream};
use crate::error::{Error, Result, TransportErrorKind};
use crate::types::{ClusterStatusMessage, StatusMessage, WorkerStatusMessage};

/// Which status endpoint to watch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WatchMode {
    /// Probe `/health` and pick the endpoint that matches.
    #[default]
    Auto,
    /// A gateway's cluster-wide broadcast.
    Cluster,
    /// A single worker's own broadcast.
    Worker,
}

impl WatchMode {
    fn path(self) -> &'static str {
        match self {
            Self::Cluster | Self::Auto => "/ws/cluster-status",
            Self::Worker => "/ws/status",
        }
    }
}

impl Client {
    /// Stream status broadcasts until the connection closes.
    ///
    /// With [`WatchMode::Auto`], `/health` is probed once to decide which endpoint to open.
    pub async fn watch(&self, mode: WatchMode) -> Result<ChunkStream<StatusMessage>> {
        let resolved = match mode {
            WatchMode::Auto => {
                // A worker's /health names itself; anything else is treated as a gateway.
                match self.health().await {
                    Ok(health) if health.kind == "worker" => WatchMode::Worker,
                    _ => WatchMode::Cluster,
                }
            }
            explicit => explicit,
        };

        let url = self.websocket_url(resolved.path())?;
        let mut request = url.as_str().into_client_request().map_err(|err| {
            Error::invalid(format!("could not build the WebSocket request: {err}"))
        })?;

        // The WebSocket handshake is an ordinary HTTP request, so it carries the same
        // credentials, under the same origin rule.
        if let Some(authorization) = self.authorization_header() {
            request
                .headers_mut()
                .insert(reqwest::header::AUTHORIZATION.as_str(), authorization);
        }
        if self.websocket_matches_base_origin(&url) {
            for (name, value) in self.edge_headers() {
                if let (Ok(name), Ok(value)) = (
                    tokio_tungstenite::tungstenite::http::HeaderName::from_bytes(
                        name.as_str().as_bytes(),
                    ),
                    HeaderValue::from_bytes(value.as_bytes()),
                ) {
                    request.headers_mut().insert(name, value);
                }
            }
        }

        let (socket, response) =
            tokio_tungstenite::connect_async(request)
                .await
                .map_err(|error| match &error {
                    tokio_tungstenite::tungstenite::Error::Http(response) => Error::Request {
                        message: format!("WebSocket connection failed: {}", response.status()),
                        code: None,
                        status: response.status().as_u16(),
                        request: None,
                    },
                    _ => Error::connection(
                        TransportErrorKind::Connect,
                        format!("could not open {url}: {error}"),
                        error,
                    ),
                })?;
        drop(response);

        Ok(Box::pin(async_stream::try_stream! {
            let mut socket = socket;
            while let Some(frame) = socket.next().await {
                let frame = frame.map_err(|error| {
                    Error::connection(
                        TransportErrorKind::MidFlight,
                        format!("status stream closed: {error}"),
                        error,
                    )
                })?;
                let payload = match frame {
                    Message::Text(text) => text.to_string(),
                    Message::Binary(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                    // Ping and pong are answered by the library; a close ends the stream.
                    Message::Close(_) => return,
                    _ => continue,
                };
                yield decode_status(&payload, resolved)?;
            }
        }))
    }

    /// The `ws`/`wss` counterpart of a path on this client's base URL.
    fn websocket_url(&self, path: &str) -> Result<Url> {
        let mut url = self.url(path)?;
        let scheme = match url.scheme() {
            "https" => "wss",
            "http" => "ws",
            other => {
                return Err(Error::invalid(format!(
                    "base_url scheme {other:?} has no WebSocket counterpart"
                )));
            }
        };
        url.set_scheme(scheme)
            .map_err(|()| Error::invalid("could not derive the WebSocket URL"))?;
        Ok(url)
    }
}

/// The endpoint decides which shape a payload is: the two share no required field.
fn decode_status(payload: &str, mode: WatchMode) -> Result<StatusMessage> {
    match mode {
        WatchMode::Worker => serde_json::from_str::<WorkerStatusMessage>(payload)
            .map(|message| StatusMessage::Worker(Box::new(message))),
        WatchMode::Cluster | WatchMode::Auto => {
            serde_json::from_str::<ClusterStatusMessage>(payload)
                .map(|message| StatusMessage::Cluster(Box::new(message)))
        }
    }
    .map_err(|err| Error::decode(format!("malformed status message: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn websocket_urls_follow_the_base_url_scheme() {
        let secure = Client::new("https://sie.example.com").unwrap();
        assert_eq!(
            secure.websocket_url("/ws/cluster-status").unwrap().as_str(),
            "wss://sie.example.com/ws/cluster-status"
        );

        let plain = Client::new("http://localhost:8080").unwrap();
        assert_eq!(
            plain.websocket_url("/ws/status").unwrap().as_str(),
            "ws://localhost:8080/ws/status"
        );
    }

    #[test]
    fn modes_map_to_their_endpoints() {
        assert_eq!(WatchMode::Cluster.path(), "/ws/cluster-status");
        assert_eq!(WatchMode::Worker.path(), "/ws/status");
        assert_eq!(WatchMode::default(), WatchMode::Auto);
    }

    #[test]
    fn payloads_decode_into_the_shape_the_endpoint_promises() {
        let worker = decode_status(
            r#"{"timestamp": 1.0, "ready": true, "name": "w-1", "machine_profile": "l4",
                "saturated": false, "gpus": [{"device": "cuda:0", "utilization_pct": 42}]}"#,
            WatchMode::Worker,
        )
        .unwrap();
        let worker = worker.worker().expect("a worker message");
        assert_eq!(worker.name, "w-1");
        assert_eq!(worker.gpus[0].utilization_pct, 42);
        assert!(worker.pool_name.is_empty());

        let cluster = decode_status(
            r#"{"timestamp": 2.0, "cluster": {"worker_count": 3, "gpu_count": 6,
                "models_loaded": 2, "total_qps": 1.5}, "workers": [], "models": []}"#,
            WatchMode::Cluster,
        )
        .unwrap();
        assert_eq!(cluster.cluster().unwrap().cluster.worker_count, 3);
        assert!(cluster.worker().is_none());
    }

    #[test]
    fn a_malformed_payload_is_a_decode_error() {
        let err = decode_status("{not json", WatchMode::Cluster).unwrap_err();
        assert!(
            err.to_string().contains("malformed status message"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn watching_an_unreachable_server_fails_rather_than_hanging() {
        let client = Client::new("http://127.0.0.1:1").unwrap();
        let Err(err) = client.watch(WatchMode::Cluster).await else {
            panic!("connecting to a closed port must fail");
        };
        assert!(matches!(err, Error::Connection { .. }), "{err:?}");
    }
}
