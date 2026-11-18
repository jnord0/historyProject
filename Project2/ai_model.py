"""
Ask GPT models (OpenAI API) and local models (Ollama) to rank historical
analogies for modern events.

Every model gets the SAME test events and the SAME shuffled candidate lists,
so the results are directly comparable with each other, with the baseline,
and later with your human labels.

Run from your Project2 folder:
    pip3 install openai pandas pyarrow
    export OPENAI_API_KEY=sk-...           (never put the key in this file)

    python3 run_models.py --check          # 1. confirm each model ID works (a few cents)
    python3 run_models.py --show-prompt    # 2. print one prompt, no API calls
    python3 run_models.py --limit 3        # 3. tiny test: 3 events per model
    python3 run_models.py                  # 4. full run (resumes if interrupted)

Outputs (in results/):
    test_set.csv          the modern events used for testing
    candidates.jsonl      the candidate list shown for each event (same for every model)
    ai_responses.jsonl    one line per (model, event): raw reply + parsed ranking + tokens
    ai_rankings.csv       tidy table: model, event, rank, historical event
"""

import os
import json
import time
import random
import argparse
import socket
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Load OPENAI_API_KEY (and HF_TOKEN etc.) from a .env file in this folder, if there is one.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# Model IDs to compare, with any extra API options for each.
# Check current IDs at https://developers.openai.com/api/docs/models and run --check.
#   GPT-4 generation: the original gpt-4 is shut down; gpt-4.1 is deprecated but may still work.
#   Reasoning models accept "reasoning"; non-reasoning ones (gpt-4.1) reject it.
#
# Local models run through Ollama (https://ollama.com). Mark them with "provider": "ollama".
# Ollama must be running and the model pulled first (ollama pull qwen3.5:4b).
#   think    True/False turns the model's thinking step on/off (only for models that support it)
#   options  Ollama settings: num_ctx = context window in tokens (prompts here are about 2,500)
#   workers  parallel calls for this model; a local GPU handles about one at a time, so keep it 1
MODELS = {
    "gpt-4o-mini": {},                                # GPT-4 generation, small tier (no reasoning option)
    "gpt-5-mini": {"reasoning": {"effort": "low"}},        # GPT-5, full model
    "gpt-6-luna": {"reasoning": {"effort": "low"}},   # GPT-6 generation, small/efficient tier
    "qwen3.5:4b": {"provider": "ollama", "think": False,      # local open model, thinking OFF
                   "options": {"num_ctx": 8192}, "workers": 1},
    # Alternatives: "gpt-5-mini" (fairer match for 4o-mini), "gpt-6.1-sol", "gpt-6-astra"
}
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = 600  # seconds to wait for one local reply

N_TEST_EVENTS = 45    # modern events to test on (spread across event types)
MIN_PER_TYPE = 3      # every event type gets at least this many test events
N_CANDIDATES = 15     # historical candidates shown per event
SEED = 42             # fixed so the test set and candidate lists never change
WORKERS = 4           # parallel API calls per model (all models run at the same time)
MODERN_CHARS = 1500   # how much of the modern event's summary to show
CANDIDATE_CHARS = 450 # how much of each candidate's summary to show

RESULTS = "results"
TEST_SET = os.path.join(RESULTS, "test_set.csv")
CANDIDATES = os.path.join(RESULTS, "candidates.jsonl")
RESPONSES = os.path.join(RESULTS, "ai_responses.jsonl")
RANKINGS = os.path.join(RESULTS, "ai_rankings.csv")

INSTRUCTIONS = """
You will be given one modern event and a numbered list of historical events.
your job is to analyze them from only info given to you; 
this can be anything from context, possible meanings/implications, a timeline of the event and its causes and effects, 
dynamics or actors involved, and to gather other details that are beyond surface level. 

You are not allowed to access the web, rely on any historical data you were trained on, or any other external sources.
It is acceptable if nothing is similar, you do not have to force a match. that being said ensure you thoroughly analyze the events.

when comparing the two events, use the details you find 
when digging into the events to rank ALL of the
historical events from most to least similar to the modern event.
A simple connection like sharing a name is not worth much especially on its own.
If none of the historical events is a genuine match, set no_good_match to true.
"""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def find(name):
    for p in (os.path.join("data", name), name, os.path.join(RESULTS, name)):
        if os.path.exists(p):
            return p
    raise SystemExit(f"Can't find {name}. Run this from your Project2 folder.")


def load_events():
    ev = pd.concat([pd.read_parquet(find("events_historical.parquet")),
                    pd.read_parquet(find("events_modern.parquet"))])
    ev = ev.drop_duplicates("qid").set_index("qid")
    return ev


def short(text, n):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def build_test_set(combined):
    """Pick N_TEST_EVENTS modern events, spread across event types. Saved once, then reused."""
    if os.path.exists(TEST_SET):
        test = pd.read_csv(TEST_SET)
        if len(test) != N_TEST_EVENTS:
            raise SystemExit(
                f"{TEST_SET} has {len(test)} events but N_TEST_EVENTS is {N_TEST_EVENTS}.\n"
                "To build a new test set, delete these files (they all depend on the test set) and rerun:\n"
                f"  {TEST_SET}\n  {CANDIDATES}\n  {RESPONSES}\n  {RANKINGS}\n"
                f"  {os.path.join(RESULTS, 'eval_scores.parquet')}")
        return test
    m = combined[["modern_qid", "modern_title", "modern_year"]].drop_duplicates("modern_qid")
    ev = load_events()
    m["event_type"] = m["modern_qid"].map(ev["event_type"]).fillna("unknown").str.split(";").str[0]
    n = min(N_TEST_EVENTS, len(m))
    groups = dict(list(m.groupby("event_type")))
    sizes = {t: len(g) for t, g in groups.items()}
    # 1) guarantee MIN_PER_TYPE of every type (or all of it, if a type has fewer)
    take = {t: min(s, MIN_PER_TYPE) for t, s in sizes.items()}
    if sum(take.values()) > n:
        raise SystemExit(f"N_TEST_EVENTS={n} is too small for {MIN_PER_TYPE} of each of "
                         f"{len(take)} event types (needs {sum(take.values())}).")
    # 2) share the rest out in proportion to how common each type is (largest remainder)
    left = n - sum(take.values())
    spare = {t: sizes[t] - take[t] for t in sizes}
    if left and sum(spare.values()):
        quota = {t: left * spare[t] / sum(spare.values()) for t in sizes}
        for t in quota:
            take[t] += int(quota[t])
        for t in sorted(quota, key=lambda t: (-(quota[t] - int(quota[t])), t)):
            if sum(take.values()) >= n:
                break
            if take[t] < sizes[t]:
                take[t] += 1
    test = pd.concat([groups[t].sample(n=take[t], random_state=SEED) for t in sorted(groups)])
    test = test.sort_values("modern_qid").reset_index(drop=True)
    os.makedirs(RESULTS, exist_ok=True)
    test.to_csv(TEST_SET, index=False)
    print(f"test set: {len(test)} events -> {TEST_SET}")
    print(test["event_type"].value_counts().to_string())
    return test


def build_candidates(test, combined, rankings, ev):
    """Same shuffled candidate list per event for every model. Saved once, then reused."""
    if os.path.exists(CANDIDATES):
        with open(CANDIDATES, encoding="utf-8") as f:
            return {c["modern_qid"]: c for c in map(json.loads, f)}
    rng = random.Random(SEED)
    historical_pool = sorted(set(rankings["historical_qid"]))
    out = {}
    for qid in test["modern_qid"]:
        c = combined[combined["modern_qid"] == qid]
        r = rankings[rankings["modern_qid"] == qid]
        chosen = {}   # historical_qid -> source (kept hidden from the models)

        def add(qids, source, limit):
            for h in qids:
                if len([1 for s in chosen.values() if s == source]) >= limit:
                    break
                chosen.setdefault(h, source)

        add(c.sort_values("combined_rank")["historical_qid"], "baseline_combined", 5)
        add(c.sort_values("keyword_rank")["historical_qid"], "baseline_keyword", 3)
        add(c.sort_values("meaning_rank")["historical_qid"], "baseline_meaning", 3)
        # harder distractors: related but outside the baseline's top 20
        mid = r[(r["embed_rank"] > 20) & (r["embed_rank"] <= 50)]["historical_qid"].tolist()
        rng.shuffle(mid)
        add(mid, "distractor_related", 2)
        # easy distractors: completely random historical events
        rand = [h for h in rng.sample(historical_pool, 30) if h not in chosen]
        add(rand, "distractor_random", 2)
        # top up from the rest of the combined ranking if anything overlapped
        for h in c.sort_values("combined_rank")["historical_qid"]:
            if len(chosen) >= N_CANDIDATES:
                break
            chosen.setdefault(h, "baseline_combined")

        items = list(chosen.items())[:N_CANDIDATES]
        rng.shuffle(items)
        out[qid] = {
            "modern_qid": qid,
            "modern_title": ev.loc[qid, "title"],
            "candidates": [{"label": str(i + 1), "qid": h, "title": ev.loc[h, "title"],
                            "year": int(ev.loc[h, "year"]), "source": s}
                           for i, (h, s) in enumerate(items)],
        }
    with open(CANDIDATES, "w", encoding="utf-8") as f:
        for c in out.values():
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"candidate lists: {len(out)} -> {CANDIDATES}")
    return out


# ---------------------------------------------------------------------------
# Prompt + API call
# ---------------------------------------------------------------------------
def build_prompt(cand, ev):
    q = cand["modern_qid"]
    lines = [f"MODERN EVENT: {ev.loc[q, 'title']} ({int(ev.loc[q, 'year'])})",
             short(ev.loc[q, "abstract"], MODERN_CHARS), "", "HISTORICAL EVENTS:"]
    for c in cand["candidates"]:
        lines.append(f"[{c['label']}] {c['title']} ({year_str(c['year'])}): "
                     f"{short(ev.loc[c['qid'], 'abstract'], CANDIDATE_CHARS)}")
    return "\n".join(lines)


def year_str(y):
    return f"{-y} BC" if y < 0 else str(y)


def schema_for(cand):
    labels = [c["label"] for c in cand["candidates"]]
    return {
        "type": "object",
        "properties": {
            "ranking": {"type": "array", "items": {"type": "string", "enum": labels},
                        "description": "ALL candidate numbers, most similar first, each exactly once"},
            "top_choice_reason": {"type": "string"},
            "no_good_match": {"type": "boolean"},
            "confidence": {"type": "integer", "description": "1 (unsure) to 5 (very sure)"},
        },
        "required": ["ranking", "top_choice_reason", "no_good_match", "confidence"],
        "additionalProperties": False,
    }


def split_config(config):
    """MODELS entry -> (provider, workers, options sent to the API)."""
    opts = dict(config)
    provider = opts.pop("provider", "openai")
    workers = opts.pop("workers", WORKERS)
    return provider, workers, opts


def call_ollama(model, options, system, prompt, schema=None):
    """One chat call to a local Ollama model. Returns an object shaped like an OpenAI
    response (.output_text, .usage.input_tokens/.output_tokens) so parse() works unchanged."""
    body = {"model": model, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            **options}
    if schema:
        body["format"] = schema   # Ollama forces the reply to match this JSON schema
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as r:
                data = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:300]
            if err.code < 500 or attempt == 2:
                raise RuntimeError(f"Ollama HTTP {err.code}: {detail}") from None
        except urllib.error.URLError as err:
            if isinstance(err.reason, (TimeoutError, socket.timeout)) and attempt < 2:
                print(f"  {model}: timed out, retrying")
                continue
            raise RuntimeError(f"Ollama is not running or not reachable at {OLLAMA_URL} "
                               f"({err.reason}). Start Ollama and rerun.") from None
        except (TimeoutError, socket.timeout):
            if attempt == 2:
                raise
            print(f"  {model}: timed out, retrying")
        time.sleep(5)
    return SimpleNamespace(
        output_text=data["message"]["content"],
        usage=SimpleNamespace(input_tokens=data.get("prompt_eval_count"),
                              output_tokens=data.get("eval_count")))


def ollama_prompt_and_schema(cand, ev):
    """Prompt + JSON schema for a local model, giving it the same information the GPT models get.
    The OpenAI API shows the model the schema's field descriptions (e.g. confidence is 1-5,
    rank ALL candidates); Ollama only enforces the JSON shape and never shows them. So:
      - append the schema to the prompt, as Ollama's docs recommend, and
      - require exactly one ranking entry per candidate (no empty or short rankings)."""
    schema = schema_for(cand)
    n = len(cand["candidates"])
    schema["properties"]["ranking"].update(minItems=n, maxItems=n)
    schema["properties"]["confidence"].update(minimum=1, maximum=5)
    prompt = (build_prompt(cand, ev) + "\n\nRespond with JSON that matches this schema:\n"
              + json.dumps(schema, indent=2))
    return prompt, schema


def call_model(client, model, config, cand, ev):
    provider, _, options = split_config(config)
    if provider == "ollama":
        return call_ollama(model, options, INSTRUCTIONS, *ollama_prompt_and_schema(cand, ev))
    kwargs = dict(
        model=model,
        instructions=INSTRUCTIONS,
        input=build_prompt(cand, ev),
        text={"format": {"type": "json_schema", "name": "ranking",
                         "schema": schema_for(cand), "strict": True}},
        **options,
    )
    for attempt in range(6):
        try:
            resp = client.responses.create(**kwargs)
            break
        except Exception as err:
            name = type(err).__name__
            retryable = name in ("RateLimitError", "APIConnectionError", "APITimeoutError",
                                 "InternalServerError")
            if not retryable or attempt == 5:
                raise
            wait = 5 * 2 ** attempt
            print(f"  {model}: {name}, retrying in {wait}s")
            time.sleep(wait)
    return resp


def parse(resp, cand):
    by_label = {c["label"]: c for c in cand["candidates"]}
    raw = resp.output_text
    rec = {"raw_text": raw, "parse_ok": False, "ranking_labels": [], "missing_labels": [],
           "no_good_match": None, "top_choice_reason": None, "confidence": None}
    try:
        data = json.loads(raw)
    except Exception:
        return rec
    seen = []
    for lab in data.get("ranking", []):
        if lab in by_label and lab not in seen:
            seen.append(lab)
    if not seen:
        return rec   # an empty ranking is a failed answer, not a ranking (it gets retried next run)
    missing = [l for l in by_label if l not in seen]
    rec.update(parse_ok=True, ranking_labels=seen + missing, missing_labels=missing,
               no_good_match=data.get("no_good_match"),
               top_choice_reason=data.get("top_choice_reason"),
               confidence=data.get("confidence"))
    rec["ranking_qids"] = [by_label[l]["qid"] for l in rec["ranking_labels"]]
    return rec


# ---------------------------------------------------------------------------
def done_pairs():
    if not os.path.exists(RESPONSES):
        return set()
    with open(RESPONSES, encoding="utf-8") as f:
        return {(r["model"], r["modern_qid"]) for r in map(json.loads, f) if r.get("parse_ok")}


def write_rankings():
    if not os.path.exists(RESPONSES):
        return
    with open(RESPONSES, encoding="utf-8") as f:
        recs = [r for r in map(json.loads, f) if r.get("parse_ok")]
    # keep the latest successful reply per (model, event)
    latest = {(r["model"], r["modern_qid"]): r for r in recs}
    with open(CANDIDATES, encoding="utf-8") as f:
        cands = {c["modern_qid"]: c for c in map(json.loads, f)}
    rows = []
    for (model, q), r in latest.items():
        info = {c["qid"]: c for c in cands[q]["candidates"]}
        for rank, h in enumerate(r["ranking_qids"], 1):
            rows.append({"model": model, "modern_qid": q, "modern_title": cands[q]["modern_title"],
                         "rank": rank, "historical_qid": h, "historical_title": info[h]["title"],
                         "historical_year": info[h]["year"], "no_good_match": r["no_good_match"],
                         "confidence": r["confidence"]})
    pd.DataFrame(rows).sort_values(["model", "modern_qid", "rank"]).to_csv(RANKINGS, index=False)
    print(f"saved {RANKINGS} ({len(latest)} model x event results)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="test each model ID with one tiny call")
    ap.add_argument("--show-prompt", action="store_true", help="print one prompt and exit")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N test events")
    ap.add_argument("--models", nargs="*", help="only run these models (default: all in MODELS)")
    ap.add_argument("--redo", action="store_true",
                    help="rerun events these models already answered (the new answers replace the old)")
    args = ap.parse_args()
    models = {m: MODELS.get(m, {}) for m in (args.models or MODELS)}

    combined = pd.read_parquet(find("baseline_combined.parquet"))
    rankings = pd.read_parquet(find("baseline_rankings.parquet"))
    ev = load_events()
    test = build_test_set(combined)
    cands = build_candidates(test, combined, rankings, ev)
    order = [q for q in test["modern_qid"] if q in cands]
    if args.limit:
        order = order[:args.limit]

    if args.show_prompt:
        c = cands[order[0]]
        print("=== INSTRUCTIONS ===\n" + INSTRUCTIONS + "\n\n=== INPUT ===\n" + build_prompt(c, ev))
        print("\n(hidden from the model) candidate sources:")
        for x in c["candidates"]:
            print(f"  [{x['label']}] {x['source']}")
        return

    # The OpenAI key is only needed when at least one OpenAI model is being run.
    client = None
    if any(split_config(c)[0] == "openai" for c in models.values()):
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set your key first: put OPENAI_API_KEY=sk-... in Project2/.env "
                             "(or run only local models with --models qwen3.5:4b)")
        from openai import OpenAI
        client = OpenAI()

    if args.check:
        for model, config in models.items():
            provider, _, opts = split_config(config)
            try:
                if provider == "ollama":
                    r = call_ollama(model, opts, "You are a helpful assistant.", "Reply with the word OK.")
                else:
                    r = client.responses.create(model=model, input="Reply with the word OK.", **opts)
                print(f"  OK    {model}: {r.output_text.strip()[:20]!r}")
            except Exception as err:
                print(f"  FAIL  {model}: {type(err).__name__}: {str(err)[:200]}")
        return

    # --redo ignores earlier answers; write_rankings() keeps only the latest answer per pair
    done = set() if args.redo else done_pairs()
    file_lock = threading.Lock()   # one writer at a time for ai_responses.jsonl
    os.makedirs(RESULTS, exist_ok=True)

    def run_model(model, opts):
        """Run every test event for one model with its own pool of threads (WORKERS,
        or the model's "workers" setting). All models run at the same time; each model
        has its own rate limit, so this is much faster than running them one by one."""
        todo = [q for q in order if (model, q) not in done]
        print(f"{model}: {len(order) - len(todo)} done already, {len(todo)} to go")
        tokens_in = tokens_out = 0

        def work(q):
            resp = call_model(client, model, opts, cands[q], ev)
            rec = parse(resp, cands[q])
            usage = getattr(resp, "usage", None)
            rec.update(model=model, modern_qid=q, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                       input_tokens=getattr(usage, "input_tokens", None),
                       output_tokens=getattr(usage, "output_tokens", None))
            with file_lock:
                with open(RESPONSES, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec

        workers = split_config(opts)[1]   # WORKERS, or the model's own "workers" setting
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, q): q for q in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    rec = fut.result()
                except Exception as err:
                    print(f"  {model}: ERROR on {futures[fut]}: {type(err).__name__}: {str(err)[:200]}")
                    msg = str(err).lower()
                    if ("model" in msg and "not" in msg) or "ollama is not running" in msg:
                        print(f"  {model}: can't run this model (wrong model ID, or Ollama not running); "
                              "skipping it")
                        for f in futures:
                            f.cancel()   # don't keep sending calls that will fail
                        break
                    continue
                tokens_in += rec["input_tokens"] or 0
                tokens_out += rec["output_tokens"] or 0
                flag = "" if rec["parse_ok"] and not rec["missing_labels"] else \
                       ("  [unparseable]" if not rec["parse_ok"] else f"  [{len(rec['missing_labels'])} missing]")
                if i % 10 == 0 or flag or i == len(todo):
                    print(f"  {model}: {i}/{len(todo)}{flag}")
        print(f"  {model}: finished, tokens this run: {tokens_in:,} in, {tokens_out:,} out")

    # One thread per model; each of those runs its own pool of WORKERS API calls.
    with ThreadPoolExecutor(max_workers=len(models)) as model_pool:
        jobs = {model_pool.submit(run_model, m, o): m for m, o in models.items()}
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception as err:
                print(f"  {jobs[job]}: stopped: {type(err).__name__}: {str(err)[:200]}")

    write_rankings()


if __name__ == "__main__":
    main()
