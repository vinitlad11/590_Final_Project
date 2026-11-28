# Adaptive QA

A cost-aware, confidence-calibrated controller for deciding when an LLM should **answer immediately**, **retrieve evidence**, or **abstain** on QANTA 2021 quizbowl questions under a computational budget.

## Current Scope (Stage 3 Corrective Diagnostic & Stage 4 Protocol)

1. **Strictly Training-Only Wikipedia Evidence Corpus (`scripts/build_wikipedia_corpus.py`):**
   - Selects the top 500 canonical Wikipedia topics strictly from `qanta.train.2021.12.20.json` (`fold == 'train'`) ranked by training-set question frequency (tie-broken by `sha256('42:' + title)`), with up to four 150-word passages per article.
   - Zero development, pilot, or test files (`selection.json`, `pilot10.jsonl`, `pilot50.jsonl`, `qanta.dev.2021.12.20.json`) are opened or consulted during corpus construction (`pilot_or_dev_files_consulted = false`).
2. **Prompting, Normalization & Model-Scale Ablation (`src/adaptive_qa/llm/prompts.py`):**
   - Prompt version `qa-answer-v4` removes literal placeholder strings (`"short answer"`), converts clue target phrases into clean category labels (`Requested entity category: painting` rather than `'this painting'`) to prevent demonstrative echoing, forbids repeating entities already named in the clue, and enforces strict single-key JSON (`{"answer": "..."}`).
   - Paired 10-question diagnostic configurations (`configs/stage04_diag10_qwen05b_traincorpus.yaml` and `configs/stage04_diag10_qwen3b_traincorpus.yaml`) isolate the effect of prompt and retrieval repairs on pinned `Qwen/Qwen2.5-0.5B-Instruct` (`7ae557604adf67be50417f59c2c2f167def9a775`) from the model upgrade to pinned `Qwen/Qwen2.5-3B-Instruct` (`aa8e72537993ba99e69dfaafa59ed015b17504d1`).
3. **Stage 4 Controller & Final Holdout Protocol (`src/adaptive_qa/controller/utility.py`, `src/adaptive_qa/pilot.py`):**
   - Implements post-hoc logistic confidence calibration (`StateCalibrator`) and expected state-action utility decision making (`LearnedUtilityController`) over actions `ANSWER`, `RETRIEVE`, and `ABSTAIN` under $U(a \mid s) = R(a \mid s) - \lambda C(a)$.
   - Evaluates 7 policies (Direct Answering, Root Answering, Always-Retrieve RAG, Fixed Confidence Threshold, Low-Confidence Triggered RAG, Learned Utility Controller, and Oracle Upper Bound) using 5-fold question-grouped cross-validation on development data before touching the independent Stage 5 final holdout set.

## Install and Test

Python 3.11 or newer is required. In your local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/run_tests.py
```

All 29 unit tests (`tests/test_*.py`) verify data splitting, leakage prevention, training-only corpus selection, BM25 title/stopword indexing, strict JSON parsing, cost/utility calculation, and collection verification.

## Main Files and Configurations

| Path | Purpose |
| :--- | :--- |
| `configs/colab_qanta2021_pilot10.yaml` | Historical Stage 2 10-question setup configuration |
| `configs/stage03_followup10.yaml` | Stage 3 10-question follow-up configuration (`Qwen2.5-3B-Instruct`, `qa-answer-v3`) |
| `configs/stage03_pilot50.yaml` | Stage 3 50-question development pilot configuration |
| `configs/stage04_diag10_qwen05b_traincorpus.yaml` | 10-question diagnostic on training-only corpus (`Qwen2.5-0.5B-Instruct`, `qa-answer-v4`) |
| `configs/stage04_diag10_qwen3b_traincorpus.yaml` | 10-question diagnostic on training-only corpus (`Qwen2.5-3B-Instruct`, `qa-answer-v4`) |
| `notebooks/STAGE_04_CORRECTIVE_DIAGNOSTIC.ipynb` | Colab notebook for training-only corpus rebuild, `0.5B` vs. `3B` diagnostic, and Stage 4 checks |
| `scripts/build_wikipedia_corpus.py` | Strictly training-only top-500 Wikipedia corpus builder |
| `scripts/run.py` | CLI supporting `prepare`, `estimate`, `baseline`, `collect`, `pilot-report`, and `evaluate` |
| `src/adaptive_qa/controller/utility.py` | Cost model, `StateCalibrator`, and `LearnedUtilityController` |

## Stage 5 Final Holdout Protocol & Compliance Rules

- **Untouched 100-Question Holdout (`scripts/prepare_stage5_holdout.py`, `configs/stage05_holdout100.yaml`):** Selects 100 untouched QANTA 2021 text-only questions (`60` untouched `train`-fold + `40` untouched `dev`-fold) strictly disjoint from all 50 Stage 3 development questions (`overlap_with_stage3_pilot50 = 0`), frozen with SHA-256 manifests before inference.
- **Zero Holdout Fitting / Frozen Single-Pass Evaluation:** The post-hoc calibrator (`StateCalibrator`), utility controller (`LearnedUtilityController`), thresholds, and hyperparameters are fit and frozen strictly on the 50-question Stage 3 development trajectories only. No fitting, cross-validation, tuning, or post-hoc modification is performed on the 100-question holdout, which is evaluated once.
- **Primary Setting vs. Sensitivity Analyses:** Primary evaluation uses `lambda = 0.1` and `R_wrong = -1.0`. Pre-specified alternatives (`lambda in {0.0, 0.05, 0.2, 0.4, 0.8}` and `R_wrong = -2.0`) are reported strictly as sensitivity analyses.
- **Complete Policy Reporting:** Every policy reports overall accuracy, selective accuracy paired with coverage/abstention rate, retrieval rate, measured token/call costs, and decision utility, alongside leave-one-group-out ablations and error analysis.
