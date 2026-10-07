# Adaptive QA

A research prototype for choosing when to answer a quizbowl question, retrieve evidence, or abstain under a computational budget.

## Current implementation

The foundation code loads text-only questions, creates partial-question observations, keeps observations from the same question in one data split, and checks answers against accepted aliases. Configuration validation and seven foundation tests are included.

The questions in `data/fixtures/` are synthetic test fixtures. They are used to check software behavior. Real QANTA model performance is measured in later experiments.

## Project structure

| Path | Purpose |
|---|---|
| `src/adaptive_qa/data/` | Question loading, prefixes and question-level splits |
| `src/adaptive_qa/evaluation/answers.py` | Answer normalization and scoring |
| `src/adaptive_qa/config.py` | Configuration defaults and validation |
| `src/adaptive_qa/io.py` | File and hashing utilities |
| `tests/test_foundations.py` | Foundation checks |
| `scripts/run_tests.py` | Test runner |
| `data/fixtures/questions.jsonl` | Synthetic questions for software checks |

## Setup and tests

Use Python 3.11 or newer. Run the following from this directory in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/run_tests.py
```

If you use Python 3.12, replace `py -3.11` with `py -3.12`. On macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_tests.py
```

A successful run ends with `Ran 7 tests` and `OK`. `requirements-tested.txt` records the direct dependency versions used to verify this package; `requirements.txt` gives the compatible ranges.

## Planned next step

The next stage adds a compact model, BM25 retrieval and a ten-question QANTA 2021 setup check on free Colab. Repeated generations will be treated as consistency signals. Formal confidence calibration is deferred until after pilot review.
