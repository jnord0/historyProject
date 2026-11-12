"""
Build two event datasets (historical < 2000, modern >= 2000) from
Wikimedia's Structured Wikipedia dataset on Hugging Face, using Wikidata
to decide which articles are events.

Step 1: query Wikidata for notable items that are instances of event classes
        (war, battle, famine, ...) and that have an English Wikipedia article.
        Results are cached in wikidata_events.csv -- delete it to re-query.
Step 2: read only the needed columns of the Hugging Face Parquet files with
        DuckDB and keep articles whose main_entity.identifier (Wikidata QID)
        is in that set.
Step 3: split by year into historical / modern files.

pip3 install requests pandas pyarrow duckdb huggingface_hub
export HF_TOKEN=hf_...   (before running)
"""

import os
import re
import time
import requests
import pandas as pd
import duckdb
from huggingface_hub import list_repo_files

# ---------------------------------------------------------------------------
# 1. Event classes to pull from Wikidata.
#    Verify each QID on wikidata.org before relying on it, and add your own
#    (search the class name on wikidata.org and copy the Q-number).
# ---------------------------------------------------------------------------
EVENT_CLASSES = {
    "war": "Q198",
    "battle": "Q178561",
    "famine": "Q168247",
    "earthquake": "Q7944",
    "revolution": "Q10931",
    "election": "Q40231",
    "pandemic": "Q12184",
    "terrorist attack": "Q2223653",
    "protest": "Q273120",
    "coup d'etat": "Q45382",
}

# Keep only events with articles in at least this many Wikipedia languages.
# Higher = fewer, more notable events. Set to 0 to keep everything.
MIN_SITELINKS = 10

WDQS = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "EventDatasetStudentProject/0.1 (class project)"}

# Date priority: point in time (P585) > start time (P580) > inception (P571)
QUERY = """
SELECT ?item ?article ?links ?pit ?start ?end ?inception WHERE {
  ?item wdt:P31/wdt:P279* wd:%s .
  ?article schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> .
  ?item wikibase:sitelinks ?links .
  FILTER(?links >= %d)
  OPTIONAL { ?item wdt:P585 ?pit . }
  OPTIONAL { ?item wdt:P580 ?start . }
  OPTIONAL { ?item wdt:P582 ?end . }
  OPTIONAL { ?item wdt:P571 ?inception . }
}
"""

EVENTS_CSV = "wikidata_events.csv"


def year_of(value):
    """Wikidata dates look like '1944-06-06T00:00:00Z' or '-0490-01-01T00:00:00Z' (BCE)."""
    if not isinstance(value, str):
        return None
    m = re.match(r"^(-?)(\d+)-", value)
    if not m:
        return None
    y = int(m.group(2))
    return -y if m.group(1) else y


def fetch_class(label, qid):
    for attempt in range(3):
        r = requests.get(WDQS, params={"query": QUERY % (qid, MIN_SITELINKS), "format": "json"},
                         headers=HEADERS, timeout=120)
        if r.status_code == 200:
            break
        time.sleep(10 * (attempt + 1))  # back off on 429 / timeouts
    r.raise_for_status()
    rows = []
    for b in r.json()["results"]["bindings"]:
        get = lambda k: b.get(k, {}).get("value")
        rows.append({
            "qid": get("item").rsplit("/", 1)[-1],
            "event_type": label,
            "url": get("article"),
            "sitelinks": int(get("links") or 0),
            "year": year_of(get("pit")) or year_of(get("start")) or year_of(get("inception")),
            "end_year": year_of(get("end")),
        })
    print(f"{label:>18}: {len(rows)} rows")
    return rows


def get_event_table():
    rows = []
    for label, qid in EVENT_CLASSES.items():
        rows += fetch_class(label, qid)
        time.sleep(2)  # be polite to the public endpoint
    df = pd.DataFrame(rows)
    # One item can match several classes / have several dates: keep earliest year,
    # and join the event types together.
    df = (df.groupby("qid")
            .agg(event_type=("event_type", lambda s: ";".join(sorted(set(s)))),
                 url=("url", "first"),
                 sitelinks=("sitelinks", "max"),
                 year=("year", "min"),
                 end_year=("end_year", "max"))
            .reset_index())
    return df


# ---------------------------------------------------------------------------
# 2. Pull matching articles from Structured Wikipedia with DuckDB.
#    Only the listed columns are downloaded; the full article body
#    ("sections") is skipped because it is most of the data.
# ---------------------------------------------------------------------------
HF_REPO = "wikimedia/structured-wikipedia"
HF_CONFIG = "enwiki_namespace_0"
MIN_ABSTRACT_CHARS = 300     # "decent amount of information" threshold; tune it


def english_parquet_paths():
    """Find the English Parquet files in the repo instead of guessing the folder."""
    files = list_repo_files(HF_REPO, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    paths = [f for f in files if HF_CONFIG in f and f.endswith(".parquet")]
    if not paths:
        sample = [f for f in files if HF_CONFIG in f][:10]
        raise SystemExit(f"No .parquet files found for {HF_CONFIG}. Files look like: {sample}")
    print(f"found {len(paths)} English parquet files, e.g. {paths[0]}")
    return [f"hf://datasets/{HF_REPO}/{p}" for p in paths]


PARTS_DIR = "parts"   # one result file per Wikipedia file, so runs can resume


def pull_articles(events):
    con = duckdb.connect()
    con.execute("SET threads = 4")   # fewer parallel requests -> fewer rate-limit hits
    token = os.environ.get("HF_TOKEN")
    if token:
        con.execute(f"CREATE SECRET hf (TYPE huggingface, TOKEN '{token}')")
    else:
        print("No HF_TOKEN set -- downloads will be slower / rate limited")
    con.register("events_df", events[["qid", "event_type", "sitelinks", "year", "end_year"]])

    os.makedirs(PARTS_DIR, exist_ok=True)
    paths = english_parquet_paths()
    print("scanning Wikipedia one file at a time (this is the slow part)...")

    for i, path in enumerate(paths):
        out = os.path.join(PARTS_DIR, f"part_{i:03d}.parquet")
        if os.path.exists(out):
            continue   # already done in an earlier run
        for attempt in range(10):
            try:
                part = con.execute("""
                    SELECT e.qid, a.name AS title, a.url, e.event_type, e.sitelinks,
                           e.year, e.end_year, a.description, a.abstract, a.infoboxes
                    FROM read_parquet(?) a
                    JOIN events_df e ON a.main_entity.identifier = e.qid
                    WHERE length(a.abstract) >= ?
                """, [path, MIN_ABSTRACT_CHARS]).df()
                break
            except Exception as err:
                if "429" not in str(err):
                    raise
                print(f"  rate limited on file {i + 1}; waiting 5 minutes...")
                time.sleep(300)
        else:
            raise SystemExit("Still rate limited after many retries; rerun later to resume.")
        part.to_parquet(out, index=False)
        print(f"file {i + 1}/{len(paths)} done: {len(part)} matches")

    return pd.concat([pd.read_parquet(os.path.join(PARTS_DIR, f))
                      for f in sorted(os.listdir(PARTS_DIR))], ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Split. Rule used here: an event belongs where it STARTS.
#    So WWII -> historical, the Iraq War (2003) -> modern, year 2000 -> modern.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if os.path.exists(EVENTS_CSV):
        print(f"using cached {EVENTS_CSV} (delete it to re-query Wikidata)")
        events = pd.read_csv(EVENTS_CSV)
    else:
        events = get_event_table()
        events.to_csv(EVENTS_CSV, index=False)
    print(f"{len(events):,} unique event items; "
          f"{events['year'].isna().sum():,} have no date and will be dropped")

    articles = pull_articles(events.dropna(subset=["year"]))

    historical = articles[articles["year"] < 2000]
    modern = articles[articles["year"] >= 2000]
    historical.to_parquet("events_historical.parquet", index=False)
    modern.to_parquet("events_modern.parquet", index=False)
    print(f"historical: {len(historical):,}   modern: {len(modern):,}")
    print(articles["event_type"].value_counts().to_string())