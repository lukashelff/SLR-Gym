# SLR-Bench

NeMo Gym resources server for SLR-Bench (Scalable Logical Reasoning) RLVR tasks.

## What This Server Verifies

For each rollout, the verifier:

- Extracts a Prolog rule from assistant output using **code-block parsing only**:
  - `[RULE]...[/RULE]` (preferred)
  - fenced ` ```prolog ... ``` `
  - fenced ` ``` ... ``` `
- Runs two judges via local SWI-Prolog execution:
  - `isomorphic`
  - `base`
- Computes shaped reward (or partial reward if configured) and selects training reward via `slr_reward`.
- Logs all scoring channels:
  - `slr_bench_isomorphic`
  - `slr_bench_base`
  - `slr_bench_format`
  - `slr_reward_hacking` (`base - isomorphic`)
  - `slr_bench_solved`
  - `reward`

## Config Knobs

In `configs/slr_bench.yaml`:

- `slr_reward`: `isomorphic` or `base` (default: `isomorphic`)
- `slr_reward_function`: `shaped` or `partial` (default: `shaped`)
- `evaluation_timeout_sec`: per-judge Prolog timeout in seconds (default: `5`)

## Dataset Row Schema

Each dataset row should include:

- `responses_create_params.input`
- `validation_program`
- `evaluation_config`:
  - `positive_predicate`
  - `negative_predicate`

## Full Dataset Generation (Local)

This repo only commits small example artifacts. Generate full `train/validation/test` locally from HF:

```bash
python resources_servers/slr_bench/scripts/create_dataset.py \
  --repo-id AIML-TUDA/SLR-Bench \
  --config-name v1-All \
  --output-dir resources_servers/slr_bench/data
```

The script writes:

- `resources_servers/slr_bench/data/train.jsonl`
- `resources_servers/slr_bench/data/validation.jsonl`
- `resources_servers/slr_bench/data/test.jsonl`

## Requirements

- Python deps in `requirements.txt`
- SWI-Prolog installed and available as `swipl`

## Running Tests

```bash
cd resources_servers/slr_bench
pytest
```
