//! Stored credentials for the data stores connector jobs read and write.
//!
//! These live on the control plane, not the gateway, so they need `control_plane_url` and
//! `org` on the client builder. Caller-supplied `base_url_headers` are gateway-edge
//! credentials and are never sent here.

use reqwest::Method;
use serde_json::{Map, Value};

use crate::client::jobs::{require_connection_name, require_connection_schema_policy};
use crate::client::{Client, meta::parse_json};
use crate::error::{Error, Result};
use crate::http::{PreparedRequest, headers};
use crate::retry::RetryPolicy;
use crate::types::{Connection, ConnectionCreated, ConnectionRevoked};

/// The connections namespace. Obtain one with [`Client::connections`].
#[derive(Debug, Clone)]
pub struct Connections {
    client: Client,
}

impl Client {
    /// Operations on stored data-store credentials.
    pub fn connections(&self) -> Connections {
        Connections {
            client: self.clone(),
        }
    }
}

impl Connections {
    fn base(&self) -> Result<reqwest::Url> {
        let (control_plane, org) = self.client.control_plane()?;
        control_plane
            .join(&format!("internal/orgs/{org}/connections"))
            .map_err(|err| Error::invalid(format!("could not build the connections URL: {err}")))
    }

    /// Store a credential.
    pub fn add(
        &self,
        name: impl Into<String>,
        connection_type: impl Into<String>,
        secret: impl Into<String>,
    ) -> ConnectionAdd {
        ConnectionAdd {
            client: self.client.clone(),
            base: self.base(),
            name: name.into(),
            connection_type: connection_type.into(),
            secret: secret.into(),
            source_schema: None,
            sink_schema: None,
        }
    }

    /// List this org's connections. Secrets are never returned.
    pub async fn list(&self) -> Result<Vec<Connection>> {
        let request = PreparedRequest::new(Method::GET, self.base()?)
            .header("accept", headers::JSON_CONTENT_TYPE);
        let response = self.client.send_once(request, RetryPolicy::NONE).await?;
        if let Ok(connections) = serde_json::from_slice::<Vec<Connection>>(&response.body) {
            return Ok(connections);
        }
        let envelope: Value = parse_json(&response, "connection list")?;
        let data = envelope
            .get("connections")
            .cloned()
            .ok_or_else(|| Error::decode("connection list is missing its `connections` array"))?;
        serde_json::from_value(data)
            .map_err(|err| Error::decode(format!("malformed connection list: {err}")))
    }

    /// Revoke a connection.
    pub async fn revoke(&self, name: &str) -> Result<ConnectionRevoked> {
        let canonical = require_connection_name(name)?;
        // The base has no trailing slash, so `join` would replace its last segment.
        let url = reqwest::Url::parse(&format!("{}/{canonical}", self.base()?))
            .map_err(|err| Error::invalid(format!("could not build the connection URL: {err}")))?;
        let request =
            PreparedRequest::new(Method::DELETE, url).header("accept", headers::JSON_CONTENT_TYPE);
        let response = self.client.send_once(request, RetryPolicy::NONE).await?;
        parse_json(&response, "connection")
    }
}

/// Stores a credential. Build with [`Connections::add`].
pub struct ConnectionAdd {
    client: Client,
    base: Result<reqwest::Url>,
    name: String,
    connection_type: String,
    secret: String,
    source_schema: Option<String>,
    sink_schema: Option<String>,
}

impl ConnectionAdd {
    /// `PostgreSQL` schema connector jobs read from. Must be paired with [`Self::sink_schema`].
    pub fn source_schema(mut self, schema: impl Into<String>) -> Self {
        self.source_schema = Some(schema.into());
        self
    }

    /// `PostgreSQL` schema connector jobs write to. Must be paired with [`Self::source_schema`].
    pub fn sink_schema(mut self, schema: impl Into<String>) -> Self {
        self.sink_schema = Some(schema.into());
        self
    }

    /// Send the request.
    pub async fn send(self) -> Result<ConnectionCreated> {
        let name = require_connection_name(&self.name)?;
        let schemas = require_connection_schema_policy(
            &self.connection_type,
            self.source_schema.as_deref(),
            self.sink_schema.as_deref(),
        )?;

        let mut body = Map::new();
        body.insert(
            "type".to_string(),
            Value::String(self.connection_type.clone()),
        );
        body.insert("name".to_string(), Value::String(name));
        body.insert("secret".to_string(), Value::String(self.secret.clone()));
        if let Some((source, sink)) = schemas {
            body.insert("source_schema".to_string(), Value::String(source));
            body.insert("sink_schema".to_string(), Value::String(sink));
        }

        let request = PreparedRequest::new(Method::POST, self.base?)
            .json_headers()
            .body(serde_json::to_vec(&Value::Object(body)).unwrap_or_default());
        let response = self.client.send_once(request, RetryPolicy::NONE).await?;
        parse_json(&response, "connection")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configured() -> Client {
        Client::builder("https://sie.example.com")
            .control_plane_url("https://cp.example.com")
            .org("acme")
            .build()
            .unwrap()
    }

    #[test]
    fn the_namespace_needs_a_control_plane_and_org() {
        let bare = Client::new("https://sie.example.com").unwrap();
        assert!(bare.connections().base().is_err());
        assert_eq!(
            configured().connections().base().unwrap().as_str(),
            "https://cp.example.com/internal/orgs/acme/connections"
        );
    }

    #[tokio::test]
    async fn a_malformed_name_is_rejected_before_any_request() {
        let client = configured();
        assert!(
            client
                .connections()
                .add("../escape", "postgres", "s")
                .send()
                .await
                .is_err()
        );
        assert!(client.connections().revoke("../escape").await.is_err());
    }

    #[tokio::test]
    async fn schemas_must_be_supplied_together() {
        let err = configured()
            .connections()
            .add("warehouse", "postgres", "secret")
            .source_schema("public")
            .send()
            .await
            .unwrap_err();
        assert!(err.to_string().contains("together"), "{err}");
    }

    #[test]
    fn control_plane_requests_carry_no_edge_headers() {
        let client = Client::builder("https://sie.example.com")
            .control_plane_url("https://cp.example.com")
            .org("acme")
            .base_url_headers(std::collections::HashMap::from([(
                "Modal-Key".to_string(),
                "k".to_string(),
            )]))
            .build()
            .unwrap();
        let control_plane = client.connections().base().unwrap();
        assert!(!client.edge_headers_apply_to(&control_plane));
    }
}
