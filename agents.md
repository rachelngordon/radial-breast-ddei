# Agents

This is a starter blueprint for automating work in this repo with task-specific agents. Refine roles, interfaces, and runbooks as the system matures.

## Agent roster
- **Data Agent**: Ingests raw k-space/HDF5 drops, validates structure, updates split files, and materializes manifests that training/eval can consume.
- **Reconstruction Agent**: Runs GRASP and related recon scripts (e.g., `grasp_recon.py`, `grasp_recon_for_val.py`), tracks run configs, and deposits reconstructed volumes in `output/<exp>/recons`.
- **Training Agent**: Launches model training (`train.py`, `train_zf.py`, `train_distributed.py`) using configs in `configs/`, manages checkpoints, and records TensorBoard/log artifacts.
- **Evaluation Agent**: Executes evaluation scripts (`eval.py`, `raw_kspace_eval.py`, `raw_grasp_eval.py`), computes metrics (SSIM/LPIPS, etc.), and writes summaries under `output/<exp>/eval_results`.
- **Reporting Agent**: Aggregates metrics across experiments, snapshots plots, and pushes concise experiment cards (inputs, config hash, key numbers, links to artifacts).

## Core workflows
1. **New dataset drop** → Data Agent validates and updates splits → Reconstruction Agent produces reconstructions if needed → Training Agent runs jobs → Evaluation Agent scores → Reporting Agent publishes results.
2. **Model/config change** → Training Agent runs targeted jobs → Evaluation Agent compares against baselines → Reporting Agent highlights deltas/regressions.
3. **Quality sweep** → Evaluation Agent batches held-out evals on recent checkpoints → Reporting Agent updates leaderboards and flags regressions.

## Hand-offs and interfaces
- Shared storage roots: `data/` (raw/splits), `output/` (checkpoints, logs, evals), `configs/` (yaml configs), `grid_search_results/` (search outputs).
- Each agent writes a machine-readable receipt per run (JSON/YAML) capturing inputs, git commit, config path, seeds, and artifact pointers.
- Artifacts are addressed by `output/<exp_name>/<run_id>/...` to keep multi-run histories.
- Logging: prefer structured logs plus short human-readable summaries for quick triage.

## Guardrails and approvals
- GPU use should be scheduled to avoid contention; Training/Recon agents check for available devices before launch.
- Dangerous actions (deleting outputs, rerunning long jobs, modifying splits) require explicit human approval.
- Always pin the git commit and config file path in receipts to make results reproducible.

## Open questions
- Which dataset versions are canonical and immutable?
- What thresholds define a regression for SSIM/LPIPS/adjacency losses?
- Do we need automated alerts (e-mail/Slack) from the Reporting Agent, and what cadence?
