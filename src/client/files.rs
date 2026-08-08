//! `/v1/files`: the OpenAI-compatible file store that feeds offline batches.

use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::Method;

use crate::client::{Client, meta::parse_json};
use crate::error::{Error, Result};
use crate::http::{PreparedRequest, headers};
use crate::retry::RetryPolicy;
use crate::types::{File, FileDeleted, FileList};

/// Upload and batch operations get a longer floor than the client timeout: they move
/// whole files, not single requests.
pub(crate) const TRANSFER_TIMEOUT_FLOOR: Duration = Duration::from_mins(2);

/// Sort order for a listing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Oldest first.
    Ascending,
    /// Newest first.
    Descending,
}

impl SortOrder {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ascending => "asc",
            Self::Descending => "desc",
        }
    }
}

/// The files namespace. Obtain one with [`Client::files`].
#[derive(Debug, Clone)]
pub struct Files {
    client: Client,
}

impl Client {
    /// Operations on uploaded files.
    pub fn files(&self) -> Files {
        Files {
            client: self.clone(),
        }
    }
}

impl Files {
    /// Upload bytes already in memory.
    pub fn upload(&self, data: impl Into<Vec<u8>>) -> FileUpload {
        FileUpload {
            client: self.client.clone(),
            source: UploadSource::Bytes(data.into()),
            filename: None,
            purpose: "batch".to_string(),
        }
    }

    /// Upload a file from disk. The filename defaults to the path's basename.
    pub fn upload_path(&self, path: impl Into<PathBuf>) -> FileUpload {
        FileUpload {
            client: self.client.clone(),
            source: UploadSource::Path(path.into()),
            filename: None,
            purpose: "batch".to_string(),
        }
    }

    /// Fetch one file's metadata.
    pub async fn retrieve(&self, file_id: &str) -> Result<File> {
        let request = self
            .client
            .request(Method::GET, &format!("/v1/files/{file_id}"))?
            .header("accept", headers::JSON_CONTENT_TYPE);
        let response = self.client.send_once(request, RetryPolicy::NONE).await?;
        parse_json(&response, "file")
    }

    /// List files.
    pub fn list(&self) -> FileListRequest {
        FileListRequest {
            client: self.client.clone(),
            after: None,
            limit: None,
            order: None,
            purpose: None,
        }
    }

    /// Download a file's contents.
    pub async fn content(&self, file_id: &str) -> Result<bytes::Bytes> {
        let request = self
            .client
            .request(Method::GET, &format!("/v1/files/{file_id}/content"))?
            .header("accept", headers::JSONL_CONTENT_TYPE);
        Ok(self
            .client
            .send_once(request, RetryPolicy::NONE)
            .await?
            .body)
    }

    /// Delete a file.
    pub async fn delete(&self, file_id: &str) -> Result<FileDeleted> {
        let request = self
            .client
            .request(Method::DELETE, &format!("/v1/files/{file_id}"))?
            .header("accept", headers::JSON_CONTENT_TYPE);
        let response = self.client.send_once(request, RetryPolicy::NONE).await?;
        parse_json(&response, "file deletion")
    }
}

#[derive(Debug, Clone)]
enum UploadSource {
    Bytes(Vec<u8>),
    Path(PathBuf),
}

/// Uploads a file. Build with [`Files::upload`] or [`Files::upload_path`].
#[derive(Debug, Clone)]
pub struct FileUpload {
    client: Client,
    source: UploadSource,
    filename: Option<String>,
    purpose: String,
}

impl FileUpload {
    /// Name recorded for the file. Defaults to the path's basename, or `upload.jsonl`.
    pub fn filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    /// What the file is for. Defaults to `batch`.
    pub fn purpose(mut self, purpose: impl Into<String>) -> Self {
        self.purpose = purpose.into();
        self
    }

    /// Send the upload.
    ///
    /// The API takes the raw bytes as the request body, with the metadata in the query
    /// string; there is no multipart form anywhere in it.
    pub async fn send(self) -> Result<File> {
        let (data, default_name) = match &self.source {
            UploadSource::Bytes(data) => (data.clone(), "upload.jsonl".to_string()),
            UploadSource::Path(path) => {
                let data = std::fs::read(path).map_err(|err| {
                    Error::Io(std::io::Error::new(
                        err.kind(),
                        format!("could not read {}: {err}", path.display()),
                    ))
                })?;
                (data, basename(path))
            }
        };
        let filename = self.filename.unwrap_or(default_name);

        let mut url = self.client.url("/v1/files")?;
        url.query_pairs_mut()
            .append_pair("purpose", &self.purpose)
            .append_pair("filename", &filename);

        let request = PreparedRequest::new(Method::POST, url)
            .header("content-type", headers::JSONL_CONTENT_TYPE)
            .header("accept", headers::JSON_CONTENT_TYPE)
            .body(data);

        let response = self
            .client
            .send_with_timeout(request, RetryPolicy::NONE, TRANSFER_TIMEOUT_FLOOR)
            .await?;
        parse_json(&response, "file")
    }
}

/// A filename never carries a directory: a caller-supplied path must not become one
/// server-side.
fn basename(path: &Path) -> String {
    path.file_name().map_or_else(
        || "upload.jsonl".to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Lists files. Build with [`Files::list`].
#[derive(Debug, Clone)]
pub struct FileListRequest {
    client: Client,
    after: Option<String>,
    limit: Option<u32>,
    order: Option<SortOrder>,
    purpose: Option<String>,
}

impl FileListRequest {
    /// Start after this file id.
    pub fn after(mut self, after: impl Into<String>) -> Self {
        self.after = Some(after.into());
        self
    }

    /// Cap on the page size.
    pub fn limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Sort order.
    pub fn order(mut self, order: SortOrder) -> Self {
        self.order = Some(order);
        self
    }

    /// Only files uploaded for this purpose.
    pub fn purpose(mut self, purpose: impl Into<String>) -> Self {
        self.purpose = Some(purpose.into());
        self
    }

    /// Fetch one page, with its pagination cursors.
    pub async fn page(self) -> Result<FileList> {
        let mut url = self.client.url("/v1/files")?;
        {
            let mut query = url.query_pairs_mut();
            if let Some(after) = &self.after {
                query.append_pair("after", after);
            }
            if let Some(limit) = self.limit {
                query.append_pair("limit", &limit.to_string());
            }
            if let Some(order) = self.order {
                query.append_pair("order", order.as_str());
            }
            if let Some(purpose) = &self.purpose {
                query.append_pair("purpose", purpose);
            }
        }

        let request =
            PreparedRequest::new(Method::GET, url).header("accept", headers::JSON_CONTENT_TYPE);
        let response = self.client.send_once(request, RetryPolicy::NONE).await?;

        // Older gateways answer with a bare array; synthesize the envelope around it.
        if let Ok(files) = serde_json::from_slice::<Vec<File>>(&response.body) {
            return Ok(FileList {
                object_kind: "list".to_string(),
                first_id: files.first().map(|file| file.id.clone()),
                last_id: files.last().map(|file| file.id.clone()),
                has_more: false,
                data: files,
            });
        }
        parse_json(&response, "file list")
    }

    /// Fetch one page and return only its files.
    pub async fn send(self) -> Result<Vec<File>> {
        Ok(self.page().await?.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_contributes_only_its_basename() {
        assert_eq!(basename(Path::new("/var/data/in.jsonl")), "in.jsonl");
        assert_eq!(basename(Path::new("in.jsonl")), "in.jsonl");
        assert_eq!(basename(Path::new("/")), "upload.jsonl");
    }

    #[tokio::test]
    async fn a_missing_upload_path_fails_before_any_request() {
        let client = Client::new("https://sie.invalid").unwrap();
        let err = client
            .files()
            .upload_path("/nonexistent/sie-sdk/in.jsonl")
            .send()
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("/nonexistent/sie-sdk/in.jsonl"),
            "{err}"
        );
    }

    #[test]
    fn sort_order_renders_the_wire_tokens() {
        assert_eq!(SortOrder::Ascending.as_str(), "asc");
        assert_eq!(SortOrder::Descending.as_str(), "desc");
    }
}
