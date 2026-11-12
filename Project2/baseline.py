"""
Baseline: rank every historical event against every modern event.

  Keywords  -> TF-IDF cosine similarity            (all pairs)
  Meaning   -> sentence-embedding cosine similarity (all pairs)
  Meaning   -> BERTScore F1                         (top-N embedding candidates only)

Run from your Project2 folder:
    pip3 install scikit-learn sentence-transformers bert-score pandas pyarrow
    python3 baseline.py --sample 20      # quick test on 20 modern events
    python3 baseline.py                  # full run

Outputs:
    features/embeddings_historical.npy, features/embeddings_modern.npy, features/qid_order.json
    results/baseline_rankings.parquet    one row per (modern event, candidate)
    results/baseline_top5.csv            human-readable top 5 per modern event
"""

import os
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EXCLUDE_TYPES = {"election"}          # event types to drop; set() to keep everything
EMBED_MODEL = "all-MiniLM-L6-v2"      # fast; "all-mpnet-base-v2" is better but slower
BERT_MODEL = "distilbert-base-uncased"  # fast; "roberta-large" is BERTScore's (slow) default
TOP_K_SAVE = 50     # candidates saved per modern event (union of TF-IDF and embedding top-K)
TOP_N_BERT = 20     # embedding top-N per modern event that get BERTScore
BERT_CHUNK = 50     # modern events per BERTScore checkpoint file

FEATURES = "features"
RESULTS = "results"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def find(name):
    for p in (os.path.join("data", name), name):
        if os.path.exists(p):
            return p
    raise SystemExit(f"Can't find {name} (looked in ./data and the current folder). "
                     "Run this from your Project2 folder.")


def load_events(name):
    df = pd.read_parquet(find(name))
    if EXCLUDE_TYPES:
        types = df["event_type"].fillna("").str.split(";")
        keep = types.apply(lambda ts: not set(ts) <= EXCLUDE_TYPES)  # drop only if ALL types excluded
        df = df[keep]
    df = df.drop_duplicates("qid").reset_index(drop=True)
    # Text used by every method: title + short description + summary
    df["text"] = (df["title"].fillna("") + ". " + df["description"].fillna("") + ". "
                  + df["abstract"].fillna("")).str.strip()
    return df


def device(use_mps=False):
    # Apple's GPU backend (MPS) crashes on some macOS / PyTorch combinations with an
    # uncatchable error, so it is opt-in (--mps). CPU is slower but reliable.
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if use_mps and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# 1. TF-IDF (keywords)
# ---------------------------------------------------------------------------
def tfidf_similarity(modern, historical):
    vec = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words only: years/numbers would be noise
        sublinear_tf=True,
        min_df=2,          # ignore words that appear in only one event
        max_df=0.5,        # ignore words in more than half of events
        ngram_range=(1, 2),
    )
    vec.fit(pd.concat([modern["text"], historical["text"]]))
    m = vec.transform(modern["text"])
    h = vec.transform(historical["text"])
    # rows are L2-normalised, so the dot product is cosine similarity
    return (m @ h.T).toarray().astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Sentence embeddings (meaning), cached to features/
# ---------------------------------------------------------------------------
def embeddings(df, which, dev):
    path = os.path.join(FEATURES, f"embeddings_{which}.npy")
    order_path = os.path.join(FEATURES, "qid_order.json")
    order = {}
    if os.path.exists(order_path):
        with open(order_path) as f:
            order = json.load(f)
    if os.path.exists(path) and order.get(which) == df["qid"].tolist() \
            and order.get("model") == EMBED_MODEL:
        print(f"  using cached {path}")
        return np.load(path)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device=dev)
    emb = model.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    os.makedirs(FEATURES, exist_ok=True)
    np.save(path, emb)
    order[which] = df["qid"].tolist()
    order["model"] = EMBED_MODEL
    with open(order_path, "w") as f:
        json.dump(order, f)
    return emb


# ---------------------------------------------------------------------------
# 3. Candidate table + ranks
# ---------------------------------------------------------------------------
def full_ranks(sim):
    """rank[i, j] = position of historical j for modern i (1 = most similar)."""
    order = np.argsort(-sim, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(sim.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, sim.shape[1] + 1)
    return ranks


def candidate_table(modern, historical, tf, em):
    tf_rank, em_rank = full_ranks(tf), full_ranks(em)
    rows = []
    for i in range(len(modern)):
        cand = set(np.where(tf_rank[i] <= TOP_K_SAVE)[0]) | set(np.where(em_rank[i] <= TOP_K_SAVE)[0])
        for j in cand:
            rows.append((i, j))
    i_idx = np.array([r[0] for r in rows])
    j_idx = np.array([r[1] for r in rows])
    return pd.DataFrame({
        "modern_qid": modern["qid"].values[i_idx],
        "modern_title": modern["title"].values[i_idx],
        "modern_year": modern["year"].values[i_idx],
        "historical_qid": historical["qid"].values[j_idx],
        "historical_title": historical["title"].values[j_idx],
        "historical_year": historical["year"].values[j_idx],
        "tfidf_score": tf[i_idx, j_idx],
        "tfidf_rank": tf_rank[i_idx, j_idx],
        "embed_score": em[i_idx, j_idx],
        "embed_rank": em_rank[i_idx, j_idx],
    })


# ---------------------------------------------------------------------------
# 4. BERTScore rerank of the embedding top-N (checkpointed, resumable)
# ---------------------------------------------------------------------------
def add_bertscore(table, modern, historical, dev):
    from bert_score import BERTScorer
    text_of = dict(zip(modern["qid"], modern["text"])) | dict(zip(historical["qid"], historical["text"]))
    todo = table[table["embed_rank"] <= TOP_N_BERT][["modern_qid", "historical_qid"]]
    parts_dir = os.path.join(FEATURES, f"bertscore_parts_{BERT_MODEL.replace('/', '_')}")
    os.makedirs(parts_dir, exist_ok=True)

    scorer = None
    qids = modern["qid"].tolist()
    chunks = [qids[k:k + BERT_CHUNK] for k in range(0, len(qids), BERT_CHUNK)]
    for c, chunk in enumerate(chunks):
        # name each checkpoint by its contents so a --sample run never gets
        # mistaken for part of the full run (or vice versa)
        key = hashlib.md5((",".join(chunk) + f"|{TOP_N_BERT}").encode()).hexdigest()[:12]
        out = os.path.join(parts_dir, f"part_{key}.parquet")
        if os.path.exists(out):
            continue
        if scorer is None:
            try:
                scorer = BERTScorer(model_type=BERT_MODEL, lang="en", device=dev)
            except Exception:
                scorer = BERTScorer(model_type=BERT_MODEL, lang="en", device="cpu")
        pairs = todo[todo["modern_qid"].isin(chunk)]
        cands = [text_of[q] for q in pairs["historical_qid"]]
        refs = [text_of[q] for q in pairs["modern_qid"]]
        _, _, f1 = scorer.score(cands, refs, batch_size=32)
        pairs.assign(bertscore_f1=f1.numpy()).to_parquet(out, index=False)
        print(f"  BERTScore chunk {c + 1}/{len(chunks)} done ({len(pairs)} pairs)")

    bs = pd.concat([pd.read_parquet(os.path.join(parts_dir, f))
                    for f in sorted(os.listdir(parts_dir))], ignore_index=True)
    bs = bs[bs["modern_qid"].isin(qids)].drop_duplicates(["modern_qid", "historical_qid"])
    table = table.merge(bs, on=["modern_qid", "historical_qid"], how="left")
    table["bertscore_rank"] = (table.groupby("modern_qid")["bertscore_f1"]
                                    .rank(ascending=False, method="first"))
    return table


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="only use the first N modern events (quick test)")
    ap.add_argument("--no-bertscore", action="store_true", help="skip the slow BERTScore step")
    ap.add_argument("--mps", action="store_true",
                    help="use the Mac GPU (faster, but crashes on some setups)")
    args = ap.parse_args()

    historical = load_events("events_historical.parquet")
    modern = load_events("events_modern.parquet")
    if args.sample:
        modern = modern.sample(n=min(args.sample, len(modern)), random_state=0).reset_index(drop=True)
    print(f"{len(modern):,} modern x {len(historical):,} historical "
          f"= {len(modern) * len(historical):,} pairs (excluded types: {EXCLUDE_TYPES or 'none'})")

    dev = device(args.mps)
    print(f"device: {dev}")

    print("1/4 TF-IDF...")
    tf = tfidf_similarity(modern, historical)

    print("2/4 embeddings...")
    em_h = embeddings(historical, "historical", dev)
    if args.sample:   # don't overwrite the full modern cache with a sample
        from sentence_transformers import SentenceTransformer
        em_m = SentenceTransformer(EMBED_MODEL, device=dev).encode(
            modern["text"].tolist(), normalize_embeddings=True, convert_to_numpy=True)
    else:
        em_m = embeddings(modern, "modern", dev)
    em = (em_m @ em_h.T).astype(np.float32)

    print("3/4 building candidate table...")
    table = candidate_table(modern, historical, tf, em)

    if args.no_bertscore:
        print("4/4 BERTScore skipped")
    else:
        print(f"4/4 BERTScore on top {TOP_N_BERT} per modern event "
              f"({len(modern) * TOP_N_BERT:,} pairs)...")
        table = add_bertscore(table, modern, historical, dev)

    os.makedirs(RESULTS, exist_ok=True)
    suffix = f"_sample{args.sample}" if args.sample else ""
    table = table.sort_values(["modern_qid", "embed_rank"])
    table.to_parquet(os.path.join(RESULTS, f"baseline_rankings{suffix}.parquet"), index=False)

    # Human-readable: top 5 per modern event by each method
    lines = []
    for method in ["tfidf", "embed"] + ([] if args.no_bertscore else ["bertscore"]):
        top = table[table[f"{method}_rank"] <= 5].copy()
        top["method"] = method
        top["rank"] = top[f"{method}_rank"].astype(int)
        lines.append(top[["modern_title", "modern_year", "method", "rank",
                          "historical_title", "historical_year"]])
    pd.concat(lines).sort_values(["modern_title", "method", "rank"]).to_csv(
        os.path.join(RESULTS, f"baseline_top5{suffix}.csv"), index=False)

    # Quick look
    print("\nExamples (embedding top 3):")
    for qid in table["modern_qid"].drop_duplicates().head(5):
        t = table[(table["modern_qid"] == qid) & (table["embed_rank"] <= 3)].sort_values("embed_rank")
        print(f"  {t['modern_title'].iloc[0]}")
        for _, r in t.iterrows():
            print(f"     {int(r['embed_rank'])}. {r['historical_title']} ({int(r['historical_year'])})")
    print(f"\nsaved results/baseline_rankings{suffix}.parquet and results/baseline_top5{suffix}.csv")


if __name__ == "__main__":
    main()