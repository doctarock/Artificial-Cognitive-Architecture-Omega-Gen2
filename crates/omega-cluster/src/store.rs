use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::protocol::CognitiveResponse;
use crate::state::{CognitiveEvent, CognitiveState, StateMutation};

#[derive(Debug, Clone)]
pub struct ClusterStore {
    root: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceRecord {
    Event {
        event: CognitiveEvent,
        state_revision: u64,
    },
    Mutation {
        mutation: StateMutation,
        state_revision: u64,
    },
    ProcessorResponse {
        response: CognitiveResponse,
        state_revision: u64,
    },
}

impl ClusterStore {
    pub fn open(root: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn load_state(&self) -> anyhow::Result<Option<CognitiveState>> {
        let path = self.state_path();
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(path)?;
        Ok(Some(serde_json::from_str(&text)?))
    }

    pub fn save_state(&self, state: &CognitiveState) -> anyhow::Result<()> {
        let text = serde_json::to_string_pretty(state)?;
        fs::write(self.state_path(), text)?;
        Ok(())
    }

    pub fn append_trace(&self, record: &TraceRecord) -> anyhow::Result<()> {
        append_jsonl(self.trace_path(), record)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn state_path(&self) -> PathBuf {
        self.root.join("state.json")
    }

    fn trace_path(&self) -> PathBuf {
        self.root.join("events.jsonl")
    }
}

fn append_jsonl(path: PathBuf, record: &TraceRecord) -> anyhow::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, record)?;
    file.write_all(b"\n")?;
    Ok(())
}
