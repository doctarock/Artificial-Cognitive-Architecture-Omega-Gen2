# Omega Research Integration Audit

Date: 2026-08-17

Scope: Milestone 2 candidates only. This audit intentionally separates reusable code or weights from reusable ideas. A `REFERENCE` or `ADAPT` decision is not permission to copy unverified data or weights into Omega.

## Summary

| Candidate | Omega subsystem | Current best strategy |
| --- | --- | --- |
| Router-R1 | Executive/router | ADAPT |
| Memory-R1 | Memory management policy | REFERENCE |
| Belief-R | Epistemic evaluation/data | ADAPT |
| Belief Engine | Epistemic belief update layer | ADAPT |
| PABU | Epistemic/progress-aware belief state | REFERENCE |
| MAGELLAN | Motivation/goals | ADAPT |
| MAP | Planning | REFERENCE |
| Metacognitive monitors | Monitor/critic | REFERENCE |

## Router-R1

- Purpose: RL-trained multi-round LLM routing and aggregation.
- Architecture: LLM router chooses among downstream LLM calls and aggregation steps; the public README describes training/evaluation over multi-dataset QA routing tasks.
- Repository: https://github.com/ulab-uiuc/Router-R1
- License: Apache-2.0 in the GitHub repository.
- Weights availability: available through the Router-R1 Hugging Face collection: https://huggingface.co/collections/ulab-ai/router-r1
- Dataset availability: repository links to model/dataset collection; README notes open-sourced dataset collected for router training.
- Training method: reinforcement learning over routing decisions, with training scripts in the repository.
- Hardware requirements: Python 3.9, PyTorch 2.4.0 CUDA 12.1, vLLM, flash-attn, verl; GPU required for practical training.
- Expected Omega compatibility: conceptually high for the executive processor, but current task is model/API routing, not cognitive processor routing.
- Integration difficulty: medium. Need an Omega-specific action space (`ROUTE`, `WAIT`, `INTERRUPT`, `CONTINUE`) and reward based on cognitive trajectory quality, not answer F1 only.
- Recommended strategy: ADAPT.

Router-R1 should influence Omega's executive training loop and registry-aware routing interface. Do not drop it in directly as the executive because Omega needs resource allocation over typed cognitive services and shared state mutations, not conversational model aggregation.

## Memory-R1

- Purpose: RL-trained memory manager and answer agent for long-horizon memory-augmented LLM agents.
- Architecture: two-agent design: Memory Manager chooses `ADD`, `UPDATE`, `DELETE`, `NOOP`; Answer Agent distills relevant memory and answers.
- Repository: https://github.com/yansikuan/memory-r1/
- License: Apache-2.0 in the GitHub repository.
- Weights availability: not confirmed in the repository at audit time.
- Dataset availability: paper/results reference LoCoMo; repository currently says code is coming soon.
- Training method: PPO and GRPO variants are reported.
- Hardware requirements: likely 7B/8B-class model fine-tuning hardware; exact scripts are not available yet.
- Expected Omega compatibility: high at the operation vocabulary level, low as a direct integration today.
- Integration difficulty: medium later, high now because code/weights are not yet published.
- Recommended strategy: REFERENCE.

Memory-R1's key reusable idea is the memory-maintenance policy. Omega should keep ownership of memory storage, indexes, associations, decay, retrieval, and traceability. A future Memory-R1-derived processor should propose maintenance mutations only.

## Belief-R

- Purpose: dataset and evaluation protocol for belief revision under new evidence.
- Architecture: delta-reasoning setup with `time_t` and `time_t1`; metrics include belief-update accuracy, belief-maintain accuracy, and BREU.
- Repository: https://github.com/HLTCHKUST/belief-revision
- License: no repository license was visible in the GitHub page during audit; treat as unlicensed until verified from files or maintainers.
- Weights availability: none; this is primarily a benchmark/dataset.
- Dataset availability: repository contains the Belief-R evaluation resources.
- Training method: not a training framework; useful for evaluation and synthetic scenario design.
- Hardware requirements: benchmark can run across LMs; requirements depend on evaluated model.
- Expected Omega compatibility: high for epistemic evaluation, especially `UPDATE_BELIEF` vs `MAINTAIN_BELIEF` cases.
- Integration difficulty: low for benchmark adaptation, medium for dataset licensing review.
- Recommended strategy: ADAPT.

Belief-R should become an evaluation suite and seed for Omega epistemic synthetic data. Do not treat multiple-choice answers as Omega's belief representation; map cases into explicit evidence references and belief confidence transitions.

## Belief Engine

- Purpose: configurable and inspectable stance/belief dynamics for multi-agent LLM deliberation.
- Architecture: extracts arguments into structured memory and updates scalar stance via a log-odds rule controlled by evidence uptake and prior anchoring.
- Repository: no implementation repository was confirmed from primary sources during audit.
- Paper: https://arxiv.org/abs/2605.15343
- License: arXiv paper license only; no code license confirmed.
- Weights availability: none identified.
- Dataset availability: evaluates against DEBATE human deliberation data; direct redistributability must be checked separately.
- Training method: primarily algorithmic/configurable belief update layer, not a learned specialist model.
- Hardware requirements: light for the update layer; extraction depends on the chosen LLM.
- Expected Omega compatibility: high for runtime-owned belief updates and traceable evidence paths.
- Integration difficulty: medium because implementation must be recreated or independently implemented from the paper.
- Recommended strategy: ADAPT.

Belief Engine is a strong fit for Omega's first non-neural epistemic substrate: explicit belief confidence updates, anchoring controls, and evidence audit trails. It is not enough for all epistemic cognition, but it can constrain learned processors.

## PABU

- Purpose: compact belief-state framework for efficient long-horizon LLM agents.
- Architecture: predicts relative task progress and selectively retains useful actions/observations instead of conditioning on full history.
- Repository: https://github.com/Hunter-Jiang/Progress-Aware-Belief-Update
- Model: https://huggingface.co/HunterJiang97/PABU-Agent-8B
- License: Hugging Face model card says the model inherits LLaMA-3.1 and downstream dataset licenses; the arXiv HTML states the codebase is Apache-2.0 and dataset is GPL-2.0.
- Weights availability: PABU-Agent-8B is available on Hugging Face.
- Dataset availability: PABU-Data is available on Hugging Face; license constraints require careful handling.
- Training method: step-level supervision from PABU Dataset over action-observation environments.
- Hardware requirements: 8B-class LLaMA derivative; inference and fine-tuning require appropriate local GPU or remote serving.
- Expected Omega compatibility: moderate. The selective-retention idea is useful, but the model is an environment-acting agent, not an Omega epistemic processor.
- Integration difficulty: high if using weights due to LLaMA and dataset license inheritance; medium if only adapting the technique.
- Recommended strategy: REFERENCE.

PABU should influence Omega's compact state and retention policies. It should not replace Omega shared state or become a monolithic acting agent.

## MAGELLAN

- Purpose: metacognitive prediction of competence and learning progress for autotelic LLM agents.
- Architecture: online RL framework for goal sampling; predicts competence/learning progress across semantically related goals.
- Repository: https://github.com/flowersteam/MAGELLAN
- License: MIT in the GitHub repository.
- Weights availability: no reusable pretrained weights identified.
- Dataset availability: experiments use LittleZoo-style environments/configs; reusable data must be checked per dependency.
- Training method: online RL with goal-sampling strategies including MAGELLAN; code uses Lamorel/LittleZoo dependencies.
- Hardware requirements: GPU/HPC-oriented configs are provided.
- Expected Omega compatibility: high for Motivation/Goal processor design, especially curiosity, learning progress, competence, and goal prioritization.
- Integration difficulty: medium. The goal-space concepts transfer better than the full environment stack.
- Recommended strategy: ADAPT.

MAGELLAN is the best current candidate for Omega motivation research. Reuse its learning-progress estimator ideas and compare against simple goal heuristics before training a bespoke motivation model.

## MAP

- Purpose: map-then-act paradigm for long-horizon interactive agent reasoning.
- Architecture: global exploration, task-specific cognitive mapping, then knowledge-augmented execution.
- Repository: no official implementation was confirmed from primary sources during audit.
- Paper: https://arxiv.org/abs/2605.13037
- License: arXiv paper license only; code/data license not confirmed.
- Weights availability: none identified.
- Dataset availability: paper introduces MAP-2K trajectories, but public availability/license was not confirmed.
- Training method: prompting/framework plus trajectory training signal; details are paper-only for now.
- Hardware requirements: depends on base LLM; benchmark runs likely require frontier or strong open models.
- Expected Omega compatibility: moderate for Planner and memory/world-state construction.
- Integration difficulty: medium to high until code/data are released.
- Recommended strategy: REFERENCE.

MAP is useful as a planning principle: build an explicit environment model before acting when the task has hidden affordances. For Omega v0.1, this should become planner interface pressure, not an implementation dependency.

## Metacognitive Monitors

- Purpose: monitor cognition for difficulty, confidence, strategy selection, loops, invalid operations, and verification feedback.
- Sources:
  - MGV paper: https://arxiv.org/abs/2511.04341
  - LLM metacognition survey and paper list: https://github.com/yale-nlp/LLM-Metacognition
  - Survey paper: https://arxiv.org/abs/2607.11881
- Architecture: no single reusable implementation. MGV formalizes `Monitor -> Generate -> Verify`; the survey taxonomizes methods and benchmarks.
- License: mixed/unknown per referenced project; treat each downstream method separately.
- Weights availability: no single monitor model identified.
- Dataset availability: varies across papers.
- Training method: varies; includes prompting, calibration, self-assessment, critique, and verifier-style training.
- Hardware requirements: varies from cheap rule-based monitors to separate model calls.
- Expected Omega compatibility: high as a design vocabulary for Monitor/Critic.
- Integration difficulty: low for heuristic monitor rules, high for learned monitor training.
- Recommended strategy: REFERENCE.

Omega's first monitor should be runtime-native and rule-based: detect cognitive loops, missing evidence references, invalid state mutations, repeated failed routes, stale assumptions, and premature actions. Learned monitor models should come after trajectory logs exist.

## Cross-Cutting Integration Rules

- Keep all candidates behind the common cognitive service protocol.
- Prefer structured outputs over chat transcripts.
- Treat model weights and datasets as separately licensed artifacts.
- Build adapters only after health/capability/process interfaces are stable.
- Compare against random, heuristic, general LLM, single cognitive LLM, specialist cluster, and trained specialist cluster baselines.
- Preserve Omega runtime ownership of state mutation, evidence validation, memory IDs, event logs, and processor availability.
