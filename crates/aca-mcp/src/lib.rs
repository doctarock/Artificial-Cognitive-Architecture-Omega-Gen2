//! MCP server surface exposing the Knowledge Library and a subset of the
//! `CognitiveLoopActor`'s channels to other household agents over
//! HTTP/LAN. Structured like `aca-api`: a consumer of already-constructed
//! handles/stores, never a second path into the graph -
//! `knowledge_library_write`/`_search` only ever touch the
//! `KnowledgeLibraryStore`, never `mental_objects`.

mod server;

pub use server::{build_router, McpState};
