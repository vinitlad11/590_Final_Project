# Adaptive QA

A research prototype for deciding when to answer a quizbowl question, retrieve evidence, or abstain under a computational budget.

## Current scope

Stage 3 adds a revised-prompt follow-up and a bounded 50-question development pilot. It retains QANTA 2021 release 2021.12.20, the pinned Qwen2.5-0.5B-Instruct model, and the fixed BM25 evidence corpus from Stage 2. The new prompt asks for an answer without an example answer value or a self-rated confidence score. Two generations measure consistency under fixed sampling settings; agreement is not calibrated confidence.

The implementation includes loading and prefixes, persistent model/retrieval caches, bounded answer/retrieval collection, usage records, and descriptive reports. Controller training and evaluation of an answer/retrieve/abstain policy are later work. Waiting, deliberation, reinforcement learning, larger models, and paid inference are outside this pilot.

## Install and test

Python 3.11 or newer is required. In an existing Windows project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/run_tests.py
```

For a fresh environment, create `.venv` first with an available Python 3.11+ interpreter. On macOS/Linux use `.venv/bin/python`. The tests use synthetic fixtures, not QANTA performance results. GPU dependencies are optional: `python -m pip install -e ".[gpu]"`.

## Pilot workflow

`notebooks/STAGE_03_PILOT50.ipynb` records the Colab workflow. It uses a separate Drive directory, imports frozen Stage 2 inputs, runs the 10-question follow-up, records a review, then collects the 50-question pilot. The original Stage 2 notebook remains available as historical setup code. Detailed operating notes and correspondence are kept outside this repository.

| Setting | Limit |
|---|---|
| Questions | 10-question follow-up, then the existing 50-question development set |
| Prefixes | Half and full text, evaluated as separate observations |
| Generations | Main answer plus one repeat at root and after retrieval |
| Additional action | One BM25 retrieval |
| Evidence | Up to three 150-word passages |
| Input and output | At most 2,048 input tokens and 128 output tokens per generation |
| Direct comparison | One independently seeded answer-only generation per prefix |

A complete 50-question collection contains 100 root states, 200 nodes, 500 logical generation calls, and 100 logical retrieval calls. With unchanged settings and retained caches, the follow-up contributes 100 calls reused in the pilot; the remaining 40 questions add 400 calls. The separate baseline command reuses the same root samples. Earlier experiments, failed attempts, and changed configurations add actual work. The call budget is per command, not a lifetime cap.

## Main files

| Path | Purpose |
|---|---|
| `configs/stage03_followup10.yaml` | Same ten questions with the revised prompt |
| `configs/stage03_pilot50.yaml` | Frozen 50-question development collection |
| `notebooks/STAGE_03_PILOT50.ipynb` | Executable Colab experiment |
| `scripts/import_stage2_data.py` | Verify and copy frozen inputs without reselection |
| `scripts/verify_collection.py` | Check completeness, trees and accounting |
| `scripts/summarize_stage3_cost.py` | Sum recorded work across Stage 3 invocations |
| `scripts/export_stage3_backup.py` | Export private source/data/results backup |
| `src/adaptive_qa/llm/` | Versioned prompts, generation backends and caches |
| `src/adaptive_qa/pilot.py` | Descriptive answer, consistency and retrieval diagnostics |
| `src/adaptive_qa/sessions.py` | Command status and wall-time records |

The CLI supports `prepare`, `estimate`, `baseline`, `collect` and `pilot-report`. Real inference requires `--execute-model`. Example commands from the project root:

```bash
python scripts/run.py estimate --config configs/stage03_followup10.yaml
python scripts/run.py baseline --config configs/stage03_followup10.yaml --execute-model
python scripts/run.py collect --config configs/stage03_followup10.yaml --execute-model
python scripts/verify_collection.py --config configs/stage03_followup10.yaml
python scripts/run.py pilot-report --config configs/stage03_followup10.yaml
```

The 50-question run uses `configs/stage03_pilot50.yaml` after the follow-up review. No Stage 3 real-model results are bundled with the source.

## Outputs and interpretation

Each run contains raw generations, a source/config/input manifest, verification results, and a pilot report. `tables/pilot_by_prefix.csv` groups by the requested half/full prefix while retaining actual rounded fractions. `sample_review.csv` includes all generated samples, parser status, placeholder flags and exact-match scores. `retrieval_review.csv` includes the passages actually retrieved. Flags do not rewrite scores or accepted aliases.

Logical usage includes all samples needed for an observation even when read from cache. Session ledgers record new attempted calls and measured successful tokens. Failed or interrupted calls can have unknown token usage; missing usage is not presented as free work. Recorded command wall time includes model loading and cache reads; it differs from summed generation latency and excludes installation and data transfer.

All pilot analyses are descriptive. Pilot questions, including those in partitions named calibration or test, are development data and must not become a later untouched holdout. The fixed evidence corpus has limited topic coverage and may contain residual Wikipedia markup. Prompt behavior, retrieval relevance, scoring and runtime need inspection before controller development. A working parser alone does not establish useful answer quality.

## Repository boundaries

Only source, configuration, synthetic fixtures, tests and unexecuted notebooks belong in Git. Downloaded data, model weights, caches, real outputs, logs, backups and professor correspondence are excluded. Run manifests preserve source hashes; Colab exports also record the Git archive commit and archive digest. Changed prompts or source require a new experiment identity and preserved prior outputs.
