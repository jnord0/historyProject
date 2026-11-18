# historyProject

Can AI models find good historical analogies for modern events?

This project takes modern events (2000 and later), such as wars, coups, protests, pandemics and terrorist attacks, and asks which past event (before 2000) is the best historical parallel. Two kinds of system answer that question:

- **Baseline:** text-similarity methods. TF-IDF measures shared keywords; sentence embeddings and BERTScore measure similar meaning.
- **AI models:** GPT models (`gpt-4o-mini`, `gpt-5-mini`, `gpt-6-luna`) rank the same candidates.

Every system ranks the **same 15 candidates** for the **same 150 test events**, so the results can be compared directly.

> See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for open problems before you rerun anything.

## Pipeline

All scripts live in `Project2/` and are run from that folder, in this order:

| # | Script | What it does | Main output |
|---|---|---|---|
| 1 | `build.py` | Asks Wikidata for notable events (10+ language editions) of 10 types, gets their English Wikipedia summaries from the Hugging Face dataset `wikimedia/structured-wikipedia`, and splits them by start year | `events_historical.parquet`, `events_modern.parquet`, `wikidata_events.csv` |
| 2 | `baseline.py` | Scores every modern × historical pair with TF-IDF and sentence embeddings (`all-MiniLM-L6-v2`), then adds BERTScore for each event's top 20. Election events are left out | `results/baseline_rankings.parquet`, `features/*.npy` |
| 3 | `combine.py` | Puts the scores on the same scale and merges them: `0.3 × keyword + 0.7 × meaning` | `results/baseline_combined.parquet` |
| 4 | `ai_model.py` | Picks 150 test events spread across event types and builds a fixed, shuffled list of 15 candidates for each. Each model ranks the list using structured JSON output | `results/test_set.csv`, `candidates.jsonl`, `ai_responses.jsonl`, `ai_rankings.csv` |
| 5 | `evaluate.py` | Compares every model with the baseline methods and a random ranker, with 95% bootstrap confidence intervals | `results/eval_summary.csv`, `eval_per_event.csv`, `eval_pairwise.csv` |

### How the 15 candidates are picked
For each test event, the list mixes:
- the baseline's top picks: 5 from the combined score, 3 by keyword, 3 by meaning
- 2 **related distractors**: ranked 21–50 by embeddings, close but not top
- 2 **random distractors**: unrelated historical events, used as a sanity check

The list is shuffled with a fixed seed (42). The models never see where each candidate came from.

## Setup

Python 3.10+.

### Install packages

Install everything at once:

```bash
pip install pandas pyarrow python-dotenv openai scipy scikit-learn sentence-transformers bert-score requests duckdb huggingface_hub
```

Or install only what the scripts you'll run need:

| Script | Packages |
|---|---|
| all scripts | `pandas` `pyarrow` (to read the `.parquet` files) `python-dotenv` (loads your `.env`) |
| `ai_model.py` | `openai` |
| `evaluate.py` | `scipy` `scikit-learn`, plus `bert-score` unless you use `--no-bertscore` |
| `baseline.py` | `scikit-learn` `sentence-transformers` `bert-score` |
| `combine.py` | nothing extra |
| `build.py` | `requests` `duckdb` `huggingface_hub` |

`sentence-transformers` and `bert-score` both install **PyTorch**, a large download (several hundred MB), so expect a few minutes. The first time it runs, each one also downloads its model (`all-MiniLM-L6-v2` or `distilbert-base-uncased`).

To check AI results only, the minimum is:

```bash
pip install pandas pyarrow python-dotenv openai scipy scikit-learn
python evaluate.py --no-bertscore
```

### API keys

API keys go in environment variables or in a `.env` file inside `Project2/`. To make one, copy the template (`Copy-Item .env.example .env` on Windows, `cp .env.example .env` on Mac/Linux) and fill in your keys. `.env` is in `.gitignore`. **Never commit keys.**

```
OPENAI_API_KEY=sk-...    # needed for ai_model.py
HF_TOKEN=hf_...          # needed for build.py
```

## Running

The data, the baseline and the AI results are already committed, so you only need to rerun the steps you change.

```bash
cd Project2

python build.py                    # slow; uses wikidata_events.csv if it exists (delete it to query again)
python baseline.py --sample 20     # quick test
python baseline.py                 # full run (add --no-bertscore to skip the slow step)
python combine.py

python ai_model.py --check         # check that each model ID works (costs a few cents)
python ai_model.py --show-prompt   # print one prompt; no API calls
python ai_model.py --limit 3       # small test: 3 events per model
python ai_model.py                 # full run; picks up where it left off if interrupted
python ai_model.py --models gpt-5-mini   # run only some models

python evaluate.py                 # add --no-bertscore for a faster run
```

Notes:
- **Run every script from inside `Project2/`** in a terminal. The scripts look for `results/` and `features/` in whatever folder they're run from. VS Code's ▶ Run button usually runs them from the repo root, so they fail to find their files.
- On Windows PowerShell, paste commands one at a time, or join them with `;`.
- On Mac/Linux, use `python3` if `python` points to Python 2.
- The first full `evaluate.py` run scores about 2,250 pairs with BERTScore (a few minutes on CPU) and caches them in `results/eval_scores.parquet`.

### Important: rerunning the AI models
`ai_model.py` skips any (model, event) pair that is already in `results/ai_responses.jsonl`. To run again after changing the prompt or models, **rename or move** `ai_responses.jsonl` and `ai_rankings.csv` first. Renaming keeps the old run so you can compare the two.

`test_set.csv` and `candidates.jsonl` are made once and reused. Delete them only if you mean to change the test set, because every model's results depend on them.

## Settings you might change

| Setting | File | Default |
|---|---|---|
| `EVENT_CLASSES`, `MIN_SITELINKS`, `MIN_ABSTRACT_CHARS` | `build.py` | 10 types, 10 languages, 300 characters |
| `EXCLUDE_TYPES`, `EMBED_MODEL`, `BERT_MODEL` | `baseline.py` | `{"election"}`, MiniLM, DistilBERT |
| `KEYWORD_WEIGHT` | `combine.py` **and** `evaluate.py` (keep them the same) | 0.3 |
| `MODELS`, `N_TEST_EVENTS`, `N_CANDIDATES`, `INSTRUCTIONS` | `ai_model.py` | 3 models, 150 events, 15 candidates |

## Reading the results

`evaluate.py` reports, for each system:
- **Pick quality:** how the system's top 1 and top 3 picks score on the baseline's keyword, meaning and combined scores, as a percentile among the 15 candidates. 1.0 means it always picks the best; 0.5 is chance.
- **Agreement:** Spearman correlation, same #1 pick, and top-3 overlap with each baseline ranking.
- **Sanity check:** how often random distractors end up at #1 or in the top 3. A good system ranks them near the bottom (around 14 of 15).
- **Model behaviour:** how often each model says no candidate is a good match, and its average confidence (1–5).

**Caveat:** these numbers measure agreement with the baseline, not correctness. A model that finds a deeper analogy the text-similarity methods miss will *lose* points. Human labels are needed to decide which system actually reasons best.

## Repository layout

```
Project2/
  build.py  baseline.py  combine.py  ai_model.py  evaluate.py
  wikidata_events.csv            Wikidata query results (cached)
  events_historical.parquet      events before 2000
  events_modern.parquet          events from 2000 on
  features/                      cached sentence embeddings + qid order
  results/                       baseline rankings, test set, candidates, AI responses
KNOWN_ISSUES.md                  open problems from code review
```
