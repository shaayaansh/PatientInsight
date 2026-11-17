"""
Retrieval judge: how good are the top-k patients each search system returns for the 100 queries?

The queries are broad ("timeline of liver abscesses ..."), so many patients besides the gold one are
relevant. Recall of the single gold patient undercounts quality. Instead an LLM judge (gpt-5.6)
labels every retrieved (query, patient) pair as relevant / not relevant, and we compute:

    P@5, P@10, P@20      fraction of the top-k that is relevant          (the main numbers)
    nDCG@20              like P@20 but rewards relevant notes ranked higher
    pooled_recall@20     relevant found / relevant found by ANY judged system for that query
                         (only meaningful when comparing systems judged together, see below)
    gold_recall@1/5/20   was the gold patient in the top-k (free, no LLM)
    any_relevant@20      at least one relevant note in the top 20

Systems (pass any number with --systems):
    bm25-raw                 BM25 with the raw query as core keywords (no agent)
    dense:<model>            dense search with the raw query, e.g. dense:medembed-base
    results:<results.jsonl>  the top-k the AGENT actually saw in an orchestrator run (its own keywords)

POOLING: judgments are cached in evals/relevance_cache.jsonl keyed by (query, patient_id). A pair that
two systems both retrieve is judged once, and every later run reuses all earlier judgments. That's
why adding a new system only costs the pairs nobody has retrieved before.

Usage (run from the project root):
    python judge_retrieval.py --systems bm25-raw dense:medembed-base --limit 5      # quick test
    python judge_retrieval.py --systems bm25-raw dense:medembed-base \
        results:results/20261115-212247_gpt-6-luna_baseline/20261115-212247_gpt-6-luna_baseline.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from judge_utils import (JUDGE_MODEL, QUERIES_PATH, JsonlCache, Usage, call_json, load_notes,
                         load_prompt, read_jsonl)

EVAL_DIR = Path("evals")
CACHE_PATH = EVAL_DIR / "relevance_cache.jsonl"
PROMPT_FILE = "judge_relevance.md"
MAX_NOTE_WORDS = 3000           # very long notes are cut (only a handful are this long)

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},      # first, so the label is conditioned on the reasoning
        "relevant": {"type": "boolean"},
    },
    "required": ["rationale", "relevant"],
    "additionalProperties": False,
}


# =========================================================================== systems -> ranked lists
def get_rankings(system: str, queries: List[Dict[str, Any]], k: int) -> Dict[str, List[int]]:
    """Return {query: [top-k patient ids in rank order]} for one system spec."""
    if system == "bm25-raw":
        from bm25_search import BM25Search
        bm25 = BM25Search.load()
        return {q["query"]: [pid for pid, s in bm25.search([q["query"]], top_k=k) if s > 0]
                for q in tqdm(queries, desc=system)}

    if system.startswith("dense:"):
        from dense_search import DenseSearch
        dense = DenseSearch.load(system.split(":", 1)[1])
        return {q["query"]: [pid for pid, _ in dense.search(q["query"], top_k=k)]
                for q in tqdm(queries, desc=system)}

    if system.startswith("results:"):
        records = {r["query"]: r for r in read_jsonl(Path(system.split(":", 1)[1]))}
        missing = [q["query"] for q in queries if q["query"] not in records]
        if missing:
            print(f"[judge_retrieval] warning: {len(missing)} queries missing from {system}; they count as empty")
        # top_k_patient_ids concatenates all searches of a run; keep first occurrence, cut to k
        out = {}
        for q in queries:
            ids = records.get(q["query"], {}).get("top_k_patient_ids", [])
            out[q["query"]] = list(dict.fromkeys(ids))[:k]
        return out

    raise ValueError(f"unknown system {system!r} (expected bm25-raw, dense:<model> or results:<path>)")


def system_label(system: str) -> str:
    """Short name for tables: results:<long path> -> results:<run folder name>."""
    if system.startswith("results:"):
        return "agent:" + Path(system.split(":", 1)[1]).parent.name
    return system


# =========================================================================== judging
def pair_key(query: str, patient_id: int) -> str:
    return hashlib.sha1(f"{query}\x00{patient_id}".encode()).hexdigest()[:16]


def judge_pair(query: str, patient_id: int, note: str, instructions: str, model: str,
               effort: str, usage: Usage) -> Dict[str, Any]:
    words = note.split()
    if len(words) > MAX_NOTE_WORDS:
        note = " ".join(words[:MAX_NOTE_WORDS]) + " [... note truncated ...]"
    user_input = f"Question: {query}\n\n<note patient_id=\"{patient_id}\">\n{note.strip()}\n</note>"
    out = call_json(instructions, user_input, RELEVANCE_SCHEMA, "relevance_judgment",
                    model=model, reasoning_effort=effort, usage=usage)
    return {"query": query, "patient_id": patient_id, "relevant": bool(out["relevant"]),
            "rationale": out["rationale"], "judge_model": model}


def judge_all(pairs: List[Tuple[str, int]], cache: JsonlCache, model: str, effort: str,
              workers: int) -> Usage:
    """Judge every (query, patient) pair not already in the cache, in parallel."""
    todo = [(q, p) for q, p in dict.fromkeys(pairs) if pair_key(q, p) not in cache]
    usage = Usage()
    print(f"[judge_retrieval] {len(set(pairs)):,} unique pairs, {len(todo):,} new to judge "
          f"({len(set(pairs)) - len(todo):,} cached)")
    if not todo:
        return usage
    notes = load_notes(p for _, p in todo)
    instructions = load_prompt(PROMPT_FILE)
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge_pair, q, p, notes[p], instructions, model, effort, usage): (q, p)
                   for q, p in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="judging"):
            q, p = futures[fut]
            try:
                cache.put(pair_key(q, p), fut.result())
            except Exception as e:          # one bad pair must not kill a 3,000-call run
                failures += 1
                tqdm.write(f"[judge_retrieval] failed on patient {p}: {type(e).__name__}: {e}")
    if failures:
        print(f"[judge_retrieval] {failures} pairs failed; re-run the same command to retry just those")
    return usage


# =========================================================================== metrics
def dcg(rels: List[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def query_metrics(ranked: List[int], labels: Dict[int, bool], gold: int, pool_relevant: set,
                  k: int) -> Dict[str, Any]:
    """Metrics for one system on one query. Unjudged pairs (failed calls) count as not relevant."""
    ranked = ranked[:k]
    rels = [1 if labels.get(p, False) else 0 for p in ranked]
    m: Dict[str, Any] = {}
    for c in (5, 10, 20):
        if c <= k:
            m[f"P@{c}"] = sum(rels[:c]) / c        # missing slots (fewer than c results) count as misses
    # ideal ranking: as many relevant notes as the pool knows of (up to k), all at the top
    ideal = dcg([1] * min(k, max(len(pool_relevant), sum(rels))))
    m[f"nDCG@{k}"] = dcg(rels) / ideal if ideal > 0 else 0.0
    m[f"pooled_recall@{k}"] = (len(set(p for p, r in zip(ranked, rels) if r) & pool_relevant)
                               / len(pool_relevant)) if pool_relevant else None
    for c in (1, 5, 20):
        if c <= k:
            m[f"gold_recall@{c}"] = float(gold in ranked[:c])
    m[f"any_relevant@{k}"] = float(sum(rels) > 0)
    m["n_relevant"] = sum(rels)
    m["n_returned"] = len(ranked)
    return m


def mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description="LLM-judged retrieval evaluation")
    ap.add_argument("--systems", nargs="+", required=True,
                    help="bm25-raw | dense:<model> | results:<results.jsonl>")
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH)
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries (for a test)")
    ap.add_argument("-k", type=int, default=20)
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--effort", default="low", help="judge reasoning effort")
    ap.add_argument("--workers", type=int, default=8, help="parallel judge calls")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    queries = read_jsonl(args.queries)[:args.limit]
    gold = {q["query"]: int(q["patient_id"]) for q in queries}

    # 1. ranked lists from every system
    rankings = {system_label(s): get_rankings(s, queries, args.k) for s in args.systems}

    # 2. judge the pool: union of every system's top-k (cached pairs are free)
    cache = JsonlCache(CACHE_PATH)
    pairs = [(q, p) for ranks in rankings.values() for q, ids in ranks.items() for p in ids]
    usage = judge_all(pairs, cache, args.model, args.effort, args.workers)

    def label(q: str, p: int):
        rec = cache.get(pair_key(q, p))
        return None if rec is None else rec["relevant"]

    # 3. metrics. The pool for pooled recall is every relevant patient ANY system here retrieved.
    per_query, summary = [], {}
    for name, ranks in rankings.items():
        rows = []
        for q in queries:
            qq = q["query"]
            pool_rel = {p for r in rankings.values() for p in r[qq] if label(qq, p)}
            labels = {p: bool(label(qq, p)) for p in ranks[qq]}
            m = query_metrics(ranks[qq], labels, gold[qq], pool_rel, args.k)
            rows.append(m)
            per_query.append({"system": name, "query": qq, "gold_patient_id": gold[qq],
                              "ranked": ranks[qq], "relevant": [p for p in ranks[qq] if labels[p]], **m})
        keys = [key for key in rows[0] if key not in ("n_returned",)]
        summary[name] = {key: round(mean([r[key] for r in rows]), 4) for key in keys}

    # 4. save + print
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = EVAL_DIR / "retrieval" / (f"{stamp}_{args.tag}" if args.tag else stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_query.jsonl", "w") as f:
        for row in per_query:
            f.write(json.dumps(row) + "\n")
    json.dump({"systems": args.systems, "k": args.k, "n_queries": len(queries), "judge_model": args.model,
               "judge_effort": args.effort, "usage_this_run": usage.as_dict(), "summary": summary},
              open(out_dir / "summary.json", "w"), indent=1)

    cols = list(next(iter(summary.values())).keys())
    width = max(len(n) for n in summary) + 2
    print("\n" + "system".ljust(width) + "".join(c.rjust(17) for c in cols))
    for name, s in summary.items():
        print(name.ljust(width) + "".join(f"{s[c]:17.3f}" for c in cols))
    print(f"\njudge usage this run: {usage}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
