"""
Evaluate the GPT models against the baseline.

Every system (each GPT model, each baseline method, and a random ranker for
reference) ranks the SAME 15 candidates for the SAME test events, so all the
numbers below are directly comparable.

Run from your Project2 folder, after baseline.py, combine.py and run_models.py:
    python3 evaluate.py
    python3 evaluate.py --no-bertscore     # skip BERTScore (faster)

Needs baseline.py in the same folder (it reuses its settings and TF-IDF code).

Outputs (in results/):
    eval_summary.csv     one row per system: every metric with a 95% confidence interval
    eval_per_event.csv   the same metrics for every (system, event)
    eval_pairwise.csv    how often each pair of systems agrees
    eval_scores.parquet  keyword / meaning scores for every candidate (cached)
"""

import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import baseline as B   # reuse the exact same event loading / TF-IDF / model settings

KEYWORD_WEIGHT = 0.3   # keep this the same as KEYWORD_WEIGHT in combine.py
N_BOOT = 2000          # bootstrap resamples for confidence intervals
SEED = 42
RESULTS = "results"
SCORES_CACHE = os.path.join(RESULTS, "eval_scores.parquet")


# ---------------------------------------------------------------------------
# 1. Load what the models saw
# ---------------------------------------------------------------------------
def load_candidates():
    with open(os.path.join(RESULTS, "candidates.jsonl")) as f:
        return {c["modern_qid"]: c for c in map(json.loads, f)}


def load_ai():
    ai = pd.read_csv(os.path.join(RESULTS, "ai_rankings.csv"))
    return ai.sort_values(["model", "modern_qid", "rank"])


# ---------------------------------------------------------------------------
# 2. Keyword and meaning scores for EVERY candidate (incl. random distractors)
# ---------------------------------------------------------------------------
def candidate_scores(cands, use_bertscore):
    if os.path.exists(SCORES_CACHE):
        s = pd.read_parquet(SCORES_CACHE)
        if set(s["modern_qid"]) >= set(cands) and (not use_bertscore or s["bertscore"].notna().all()):
            print(f"using cached {SCORES_CACHE}")
            return s

    historical = B.load_events("events_historical.parquet")
    modern = B.load_events("events_modern.parquet")
    pairs = pd.DataFrame([(q, c["qid"], c["source"]) for q, v in cands.items() for c in v["candidates"]],
                         columns=["modern_qid", "historical_qid", "source"])

    print("TF-IDF (same settings as baseline.py)...")
    tf = B.tfidf_similarity(modern, historical)
    mi = {q: i for i, q in enumerate(modern["qid"])}
    hi = {q: i for i, q in enumerate(historical["qid"])}
    pairs["tfidf"] = [tf[mi[m], hi[h]] for m, h in zip(pairs["modern_qid"], pairs["historical_qid"])]

    print("embeddings (cached vectors from baseline.py)...")
    with open(os.path.join(B.FEATURES, "qid_order.json")) as f:
        order = json.load(f)
    if "modern" not in order:
        raise SystemExit("Modern embeddings not found. Run the full baseline.py first (not --sample).")
    em_m = np.load(os.path.join(B.FEATURES, "embeddings_modern.npy"))
    em_h = np.load(os.path.join(B.FEATURES, "embeddings_historical.npy"))
    emi = {q: i for i, q in enumerate(order["modern"])}
    ehi = {q: i for i, q in enumerate(order["historical"])}
    pairs["embed"] = [float(em_m[emi[m]] @ em_h[ehi[h]])
                      for m, h in zip(pairs["modern_qid"], pairs["historical_qid"])]

    pairs["bertscore"] = np.nan
    if use_bertscore:
        print(f"BERTScore on {len(pairs):,} candidate pairs (a few minutes on CPU)...")
        from bert_score import BERTScorer
        text = dict(zip(modern["qid"], modern["text"])) | dict(zip(historical["qid"], historical["text"]))
        scorer = BERTScorer(model_type=B.BERT_MODEL, lang="en", device=B.device())
        _, _, f1 = scorer.score([text[h] for h in pairs["historical_qid"]],
                                [text[m] for m in pairs["modern_qid"]], batch_size=32)
        pairs["bertscore"] = f1.numpy()

    os.makedirs(RESULTS, exist_ok=True)
    pairs.to_parquet(SCORES_CACHE, index=False)
    return pairs


def z(s):
    return (s - s.mean()) / (s.std() + 1e-9)


def add_baseline_scores(scores, use_bertscore):
    """Keyword / meaning / combined scores, standardised within each event's 15 candidates."""
    s = scores.copy()
    g = s.groupby("modern_qid")
    s["keyword"] = g["tfidf"].transform(z)
    if use_bertscore:
        s["meaning"] = (g["embed"].transform(z) + g["bertscore"].transform(z)) / 2
    else:
        s["meaning"] = g["embed"].transform(z)
    s["combined"] = KEYWORD_WEIGHT * s["keyword"] + (1 - KEYWORD_WEIGHT) * s["meaning"]
    return s


# ---------------------------------------------------------------------------
# 3. Every system's ranking of the same 15 candidates
# ---------------------------------------------------------------------------
def all_rankings(ai, scores, use_bertscore):
    """{system: {modern_qid: [historical_qid, ...] best first}}"""
    systems = {}
    for model, g in ai.groupby("model"):
        systems[model] = {q: list(gg["historical_qid"]) for q, gg in g.groupby("modern_qid")}

    methods = ["keyword", "meaning", "combined", "tfidf", "embed"] + (["bertscore"] if use_bertscore else [])
    for col in methods:
        name = f"baseline_{col}"
        systems[name] = {q: list(g.sort_values(col, ascending=False)["historical_qid"])
                         for q, g in scores.groupby("modern_qid")}

    rng = np.random.default_rng(SEED)
    systems["random_ranker"] = {q: list(rng.permutation(g["historical_qid"].values))
                                for q, g in scores.groupby("modern_qid")}
    return systems


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def percentile_in_event(values, value):
    """Share of the other candidates this value beats (0 = worst of 15, 1 = best of 15)."""
    others = len(values) - 1
    return float((values < value).sum() / others) if others else np.nan


def per_event_metrics(systems, scores, ai):
    sc = {q: g.set_index("historical_qid") for q, g in scores.groupby("modern_qid")}
    ref = {k: systems[f"baseline_{k}"] for k in ["keyword", "meaning", "combined"]}
    extra = ai[ai["rank"] == 1].set_index(["model", "modern_qid"])[["no_good_match", "confidence"]]

    rows = []
    for name, ranks in systems.items():
        for q, order in ranks.items():
            if q not in sc:
                continue
            s = sc[q]
            pos = {h: i + 1 for i, h in enumerate(order)}
            top1, top3 = order[0], order[:3]
            row = {"system": name, "modern_qid": q}

            # Pick quality, judged by the baseline's own scores (percentile within the 15)
            for col in ["keyword", "meaning", "combined"]:
                vals = s[col].values
                row[f"{col}_pct_top1"] = percentile_in_event(vals, s.loc[top1, col])
                row[f"{col}_pct_top3"] = np.mean([percentile_in_event(vals, s.loc[h, col]) for h in top3])

            # Agreement with each baseline ranking
            for k, refr in ref.items():
                if name == f"baseline_{k}":
                    continue
                r = refr[q]
                rpos = {h: i + 1 for i, h in enumerate(r)}
                common = [h for h in order if h in rpos]
                row[f"spearman_{k}"] = spearmanr([pos[h] for h in common], [rpos[h] for h in common])[0]
                row[f"top1_match_{k}"] = float(top1 == r[0])
                row[f"overlap3_{k}"] = len(set(top3) & set(r[:3])) / 3

            # Distractors: a good system should almost never rank random events highly
            src = s["source"]
            rand = [h for h in order if src.get(h) == "distractor_random"]
            row["top1_is_random"] = float(src.get(top1) == "distractor_random")
            row["random_in_top3"] = float(any(src.get(h) == "distractor_random" for h in top3))
            row["random_mean_rank"] = np.mean([pos[h] for h in rand]) if rand else np.nan
            row["top1_source"] = src.get(top1)

            if (name, q) in extra.index:
                row["no_good_match"] = float(bool(extra.loc[(name, q), "no_good_match"]))
                row["confidence"] = float(extra.loc[(name, q), "confidence"])
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(x, rng):
    x = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def summarise(per_event):
    rng = np.random.default_rng(SEED)
    metric_cols = [c for c in per_event.columns
                   if c not in ("system", "modern_qid", "top1_source")]
    rows = []
    for name, g in per_event.groupby("system"):
        row = {"system": name, "n_events": len(g)}
        for c in metric_cols:
            m, lo, hi = bootstrap_ci(g[c], rng)
            row[c], row[c + "_lo"], row[c + "_hi"] = m, lo, hi
        src = g["top1_source"].value_counts(normalize=True)
        for s in ["baseline_combined", "baseline_keyword", "baseline_meaning",
                  "distractor_related", "distractor_random"]:
            row[f"top1_from_{s}"] = src.get(s, 0.0)
        rows.append(row)
    out = pd.DataFrame(rows)
    # Headline scores from the project plan: keyword category, meaning category, and overall
    out["keyword_score"] = out["keyword_pct_top3"]
    out["meaning_score"] = out["meaning_pct_top3"]
    out["overall_score"] = (out["keyword_score"] + out["meaning_score"]) / 2
    return out


def pairwise(systems):
    names = [n for n in systems if n != "random_ranker"]
    rows = []
    for a, b in itertools.combinations(names, 2):
        qs = sorted(set(systems[a]) & set(systems[b]))
        top1 = np.mean([systems[a][q][0] == systems[b][q][0] for q in qs])
        rhos = []
        for q in qs:
            pa = {h: i for i, h in enumerate(systems[a][q])}
            pb = {h: i for i, h in enumerate(systems[b][q])}
            common = [h for h in pa if h in pb]
            rhos.append(spearmanr([pa[h] for h in common], [pb[h] for h in common])[0])
        rows.append({"system_a": a, "system_b": b, "n_events": len(qs),
                     "top1_agreement": top1, "mean_spearman": np.nanmean(rhos)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def fmt(row, col):
    return f"{row[col]:.2f} [{row[col + '_lo']:.2f}-{row[col + '_hi']:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bertscore", action="store_true", help="meaning = embeddings only (faster)")
    args = ap.parse_args()
    use_bert = not args.no_bertscore

    cands = load_candidates()
    ai = load_ai()
    ai = ai[ai["modern_qid"].isin(cands)]
    scores = add_baseline_scores(candidate_scores(cands, use_bert), use_bert)
    systems = all_rankings(ai, scores, use_bert)

    per_event = per_event_metrics(systems, scores, ai)
    summary = summarise(per_event)
    pairs = pairwise(systems)

    per_event.to_csv(os.path.join(RESULTS, "eval_per_event.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS, "eval_summary.csv"), index=False)
    pairs.to_csv(os.path.join(RESULTS, "eval_pairwise.csv"), index=False)

    models = sorted(ai["model"].unique())
    show = summary.set_index("system")
    print(f"\n{len(cands)} test events, {len(models)} models, meaning = "
          f"{'embeddings + BERTScore' if use_bert else 'embeddings only'}, keyword weight {KEYWORD_WEIGHT}")

    print("\n=== Headline: quality of each model's top 3 picks, judged by the baseline ===")
    print("(percentile among the 15 candidates: 1.00 = always the best, 0.50 = random)")
    for name in sorted(models, key=lambda m: -show.loc[m, "overall_score"]) + ["random_ranker"]:
        r = show.loc[name]
        print(f"  {name:<16} keyword {fmt(r, 'keyword_pct_top3')}   meaning {fmt(r, 'meaning_pct_top3')}"
              f"   overall {r['overall_score']:.2f}")

    print("\n=== Agreement with the combined baseline ranking ===")
    for name in models + ["random_ranker"]:
        r = show.loc[name]
        print(f"  {name:<16} spearman {fmt(r, 'spearman_combined')}   same #1 {fmt(r, 'top1_match_combined')}"
              f"   top-3 overlap {fmt(r, 'overlap3_combined')}")

    print("\n=== Sanity checks: random distractors (should be ranked near the bottom, ~14) ===")
    for name in models + ["baseline_combined", "random_ranker"]:
        r = show.loc[name]
        print(f"  {name:<18} #1 is random {r['top1_is_random']:.1%}   random in top 3 {r['random_in_top3']:.1%}"
              f"   avg rank of random {r['random_mean_rank']:.1f}")

    print("\n=== Model behaviour ===")
    for name in models:
        r = show.loc[name]
        print(f"  {name:<16} says 'no good match' {r['no_good_match']:.1%}   avg confidence {r['confidence']:.2f}")

    print("\n=== Model vs model ===")
    mp = pairs[pairs["system_a"].isin(models) & pairs["system_b"].isin(models)]
    for _, r in mp.iterrows():
        print(f"  {r['system_a']} vs {r['system_b']}: same #1 {r['top1_agreement']:.0%}, "
              f"spearman {r['mean_spearman']:.2f}")

    print("\nsaved results/eval_summary.csv, eval_per_event.csv, eval_pairwise.csv")
    print("Note: these scores measure agreement with the baseline, not correctness. "
          "Human labels are needed to say which model reasons best.")


if __name__ == "__main__":
    main()