use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use omega_cluster::store::TraceRecord;

fn main() -> anyhow::Result<()> {
    let path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("dist/omega-cluster/events.jsonl"));
    let file = File::open(&path)?;
    let reader = BufReader::new(file);

    println!("trace: {}", path.display());
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: TraceRecord = serde_json::from_str(&line)?;
        print_record(index + 1, record);
    }
    Ok(())
}

fn print_record(index: usize, record: TraceRecord) {
    match record {
        TraceRecord::Event {
            event,
            state_revision,
        } => {
            println!(
                "{index:04} r{state_revision:<4} event      {:<22} {}",
                event.kind, event.payload
            );
        }
        TraceRecord::Mutation {
            mutation,
            state_revision,
        } => {
            println!(
                "{index:04} r{state_revision:<4} mutation   {}",
                mutation_name(&mutation)
            );
        }
        TraceRecord::ProcessorResponse {
            response,
            state_revision,
        } => {
            println!(
                "{index:04} r{state_revision:<4} processor  {:<10} {:?} target={:?} confidence={:.2}",
                response.processor,
                response.result.operation,
                response.result.target_processor,
                response.confidence
            );
        }
    }
}

fn mutation_name(mutation: &omega_cluster::state::StateMutation) -> &'static str {
    match mutation {
        omega_cluster::state::StateMutation::AddWorkspaceCandidate { .. } => {
            "add_workspace_candidate"
        }
        omega_cluster::state::StateMutation::SetFocus { .. } => "set_focus",
        omega_cluster::state::StateMutation::AddWorkingMemory { .. } => "add_working_memory",
        omega_cluster::state::StateMutation::AddObservation { .. } => "add_observation",
        omega_cluster::state::StateMutation::AddRecentEvent { .. } => "add_recent_event",
        omega_cluster::state::StateMutation::RecordOperation { .. } => "record_operation",
    }
}
