# Adaptive QA

A research prototype for deciding when to answer a quizbowl question, retrieve evidence, or abstain under a computational budget.

## Current implementation

Stage 2 includes question loading and prefixes, answer parsing, persistent generation caches, BM25 retrieval, bounded collection of candidate and retrieved answers, and descriptive consistency/cost reports. There are 22 software tests. The learned controller is introduced in a later stage.

The real setup check uses QANTA 2021, `Qwen/Qwen2.5-0.5B-Instruct` and free Google Colab. Repeated generations are consistency signals. Calibration is disabled in the pilot configuration. Matching answers can still be wrong.

## Install and test

Python 3.11 or newer is required. From this project directory in PowerShell, use the environment you created in Stage 1:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/run_tests.py
```

For a fresh environment, first run `py -3.11 -m venv .venv` (or use your installed Python version). On macOS/Linux use `.venv/bin/python` in place of the Windows executable. A successful run ends with `Ran 22 tests` and `OK`.

GPU and data preparation dependencies are installed in Colab with `python -m pip install -e ".[gpu,data]"`. They are not required for the local fixture tests.

## Run the real setup check

Open `notebooks/STAGE_02_SETUP10.ipynb` in Google Colab and run cells individually. It prepares and freezes reviewed QANTA data, builds an independent Wikipedia corpus, pins the model revision, collects ten questions and produces a descriptive report. Human review steps precede inference.

The configuration uses two independent prefixes (half and full question text), two generations at each observation, and at most one retrieval. BM25 returns up to three 150-word passages. A request is limited to 2,048 input tokens and 128 output tokens. The complete ten-question collection has at most 100 unique generation requests with unchanged settings and preserved caches; failed attempts or changed settings can add actual work.

## Main files

| Path | Purpose |
|---|---|
| `configs/colab_qanta2021_pilot10.yaml` | Real ten-question configuration |
| `configs/fixture_core_check.yaml` | Synthetic software-check configuration |
| `notebooks/STAGE_02_SETUP10.ipynb` | Executable Colab workflow |
| `scripts/prepare_qanta_public.py` | Public data download, review and freeze |
| `scripts/build_wikipedia_corpus.py` | Fixed evidence corpus and source records |
| `src/adaptive_qa/llm/` | Generation backends, prompts and parsing |
| `src/adaptive_qa/retrieval/` | BM25 retrieval and evidence features |
| `src/adaptive_qa/outcomes.py` | Bounded answer and retrieval collection |
| `src/adaptive_qa/pilot.py` | Consistency and usage diagnostics |

The CLI commands for this stage are `prepare`, `estimate`, `baseline`, `collect` and `pilot-report`; `scripts/verify_collection.py` checks accounting and tree structure. Controller fitting is introduced in a later stage.

## Data and generated outputs

Files in `data/fixtures/` are synthetic software fixtures. Real data, model weights, caches and run outputs are excluded from Git. Colab keeps working data and results in Drive for resuming and backup. Keep downloaded reports, test logs and correspondence in your separate private folder.

Software checks establish pipeline behavior. Real model quality, retrieval coverage, memory use and runtime must be measured in the ten-question setup check.
