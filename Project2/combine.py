"""
Combine the baseline scores into keyword, meaning and overall rankings.

Run from your Project2 folder after baseline.py:
    python3 combine.py

Output: results/baseline_combined.parquet
"""

import pandas as pd

KEYWORD_WEIGHT = 0.3   # combined = KEYWORD_WEIGHT * keyword + (1 - KEYWORD_WEIGHT) * meaning

d = pd.read_parquet("results/baseline_rankings.parquet")
d = d[d["bertscore_f1"].notna()].copy()   # the top-20 candidates that have all three scores


def z(s):
    return (s - s.mean()) / (s.std() + 1e-9)


# Put every score on the same scale within each modern event's candidates
for col in ["tfidf_score", "embed_score", "bertscore_f1"]:
    d[col + "_z"] = d.groupby("modern_qid")[col].transform(z)

d["keyword_score"] = d["tfidf_score_z"]
d["meaning_score"] = (d["embed_score_z"] + d["bertscore_f1_z"]) / 2
d["combined_score"] = KEYWORD_WEIGHT * d["keyword_score"] + (1 - KEYWORD_WEIGHT) * d["meaning_score"]

for col in ["keyword_score", "meaning_score", "combined_score"]:
    d[col.replace("score", "rank")] = (d.groupby("modern_qid")[col]
                                         .rank(ascending=False, method="first").astype(int))

d = d.sort_values(["modern_qid", "combined_rank"])
d.to_parquet("results/baseline_combined.parquet", index=False)

# Readable version: combined top 5 per modern event
d[d["combined_rank"] <= 5][["modern_title", "modern_year", "combined_rank",
                            "historical_title", "historical_year",
                            "keyword_rank", "meaning_rank"]] \
    .to_csv("results/baseline_combined_top5.csv", index=False)

print(f"{d['modern_qid'].nunique():,} modern events combined")
print(d[d["combined_rank"] <= 3][["modern_title", "combined_rank", "historical_title"]]
      .head(15).to_string(index=False))
print("\nsaved results/baseline_combined.parquet and results/baseline_combined_top5.csv")