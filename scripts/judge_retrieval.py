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
    shown:<results.jsonl>    the notes the agent actually READ in that run (after verification). Shorter than k,
                             so read prec_all (relevant / returned) rather than P@20 for these.
    hybrid:<results.jsonl>   RRF of BM25 (re-run with the keywords the agent chose in that run, first
                             search only) and dense (raw query, --dense-model). One system per
                             combination of --w-dense x --rrf-k x --depth, labelled e.g.
                             hybrid[w=0.5,k=60,d=100]:<run>. No agent call: it reuses the run's keywords.

RELEVANCE PROMPT: queries with a gold patient_id (the temporal set) use prompts/judge_relevance.md ("does this
note help answer the question?"). Queries without one (the cohort set) use judge_relevance_cohort.md ("does this
patient meet every criterion?"). Override with --prompt.

VERIFIER AGREEMENT: for runs made with --verify, every screened note is judged too, and the verifier's keep/drop
decisions are compared with the judge's labels (precision and recall of what it kept, accuracy per label).

POOLING: judgments are cached in evals/relevance_cache.jsonl keyed by (query, patient_id). A pair that
two systems both retrieve is judged once, and every later run reuses all earlier judgments. That's
why adding a new system only costs the pairs nobody has retrieved before.

Usage (run from the project root):
    python scripts/judge_retrieval.py --systems bm25-raw dense:medembed-base --limit 5      # quick test
    python scripts/judge_retrieval.py --systems bm25-raw dense:medembed-base \
        results:results/20261115-212247_gpt-6-luna_baseline/20261115-212247_gpt-6-luna_baseline.jsonl
    python scripts/judge_retrieval.py --systems results:$RUN dense:medembed-base hybrid:$RUN \
        --w-dense 0.25 0.5 1.0 --rrf-k 20 60 --depth 100
    python scripts/judge_retrieval.py --queries data/asclepius/cohort_queries/cohort_queries.jsonl \
        --systems results:$HYB results:$VER shown:$VER --tag cohort
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repo root (this file lives in scripts/)
sys.path.insert(0, str(ROOT))                       # so the root modules (bm25_search, ...) import from here

import argparse
import functools
import hashlib
import itertools
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from judge_utils import (JUDGE_MODEL, QUERIES_PATH, JsonlCache, Usage, call_json, load_notes,
                         load_prompt, read_jsonl)

EVAL_DIR = ROOT / "evals"
CACHE_PATH = EVAL_DIR / "relevance_cache.jsonl"
PROMPT_FILE = "judge_relevance.md"                 # temporal / evidence questions (default)
COHORT_PROMPT_FILE = "judge_relevance_cohort.md"   # cohort definitions
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
@functools.lru_cache(maxsize=None)
def load_bm25():
    from bm25_search import BM25Search
    return BM25Search.load()


@functools.lru_cache(maxsize=None)
def load_dense(model: str):
    from dense_search import DenseSearch
    return DenseSearch.load(model)


def get_rankings(system: str, queries: List[Dict[str, Any]], k: int) -> Dict[str, List[int]]:
    """Return {query: [top-k patient ids in rank order]} for one system spec."""
    if system == "bm25-raw":
        bm25 = load_bm25()
        return {q["query"]: [pid for pid, s in bm25.search([q["query"]], top_k=k) if s > 0]
                for q in tqdm(queries, desc=system)}

    if system.startswith("dense:"):
        dense = load_dense(system.split(":", 1)[1])
        return {q["query"]: [pid for pid, _ in dense.search(q["query"], top_k=k)]
                for q in tqdm(queries, desc=system)}

    if system.startswith("shown:"):
        records = {r["query"]: r for r in read_jsonl(Path(system.split(":", 1)[1]))}
        return {q["query"]: list(dict.fromkeys(records.get(q["query"], {}).get("shown_patient_ids")
                                               or records.get(q["query"], {}).get("top_k_patient_ids", [])))[:k]
                for q in queries}

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

    raise ValueError(f"unknown system {system!r} "
                     f"(expected bm25-raw, dense:<model>, results:<path>, shown:<path> or hybrid:<path>)")


def hybrid_label(w: float, rk: float, depth: int, run_name: str) -> str:
    return f"hybrid[w={w:g},k={rk:g},d={depth}]:{run_name}"


def get_hybrid_rankings(results_path: Path, queries: List[Dict[str, Any]], k: int, dense_model: str,
                        settings: List[Tuple[float, float, int]]) -> Dict[str, Dict[str, List[int]]]:
    """
    One system per (w_dense, rrf_k, depth) setting. Each query is searched ONCE per retriever (to the
    largest depth) and then fused under every setting, so trying many settings is nearly free.
    BM25 re-uses the keywords the agent chose in that run, with the run's aux_weight.
    """
    from hybrid_search import HybridSearch
    config_path = results_path.parent / "config.json"
    aux_weight = json.load(open(config_path)).get("aux_weight", 0.5) if config_path.exists() else 0.5
    hs = HybridSearch(load_bm25(), load_dense(dense_model), aux_weight=aux_weight)
    records = {r["query"]: r for r in read_jsonl(results_path)}
    run_name = results_path.parent.name
    max_depth = max(d for _, _, d in settings)

    out: Dict[str, Dict[str, List[int]]] = {hybrid_label(w, rk, d, run_name): {} for w, rk, d in settings}
    no_search = 0
    for q in tqdm(queries, desc=f"hybrid:{run_name}"):
        qq = q["query"]
        searches = records.get(qq, {}).get("searches") or []
        if searches:
            s = searches[0]
            b = hs.bm25_ranked(s["core_keywords"], s.get("auxiliary_keywords") or [], depth=max_depth)
        else:                                   # the agent never searched: dense alone decides
            b, no_search = [], no_search + 1
        d = hs.dense_ranked(qq, depth=max_depth)
        for w, rk, dep in settings:
            fused = hs.fuse(b, d, top_k=k, k=rk, w_dense=w, depth=dep)
            out[hybrid_label(w, rk, dep, run_name)][qq] = [pid for pid, _ in fused]
    if no_search:
        print(f"[judge_retrieval] {no_search} queries had no agent search in {run_name}; hybrid = dense only there")
    return out


def system_label(system: str) -> str:
    """Short name for tables: results:<long path> -> results:<run folder name>."""
    if system.startswith("results:"):
        return "agent:" + Path(system.split(":", 1)[1]).parent.name
    if system.startswith("shown:"):
        return "shown:" + Path(system.split(":", 1)[1]).parent.name
    return system


# =========================================================================== judging
def pair_key(query: str, patient_id: int, prompt_file: str = PROMPT_FILE) -> str:
    """Cache key. The default prompt keeps the original key, so earlier judgments stay valid; any other
    prompt gets its own key, so labels made under different definitions never mix."""
    extra = "" if prompt_file == PROMPT_FILE else f"\x00{prompt_file}"
    return hashlib.sha1(f"{query}\x00{patient_id}{extra}".encode()).hexdigest()[:16]


def judge_pair(query: str, patient_id: int, note: str, instructions: str, model: str,
               effort: str, usage: Usage, prompt_file: str = PROMPT_FILE) -> Dict[str, Any]:
    words = note.split()
    if len(words) > MAX_NOTE_WORDS:
        note = " ".join(words[:MAX_NOTE_WORDS]) + " [... note truncated ...]"
    user_input = f"Question: {query}\n\n<note patient_id=\"{patient_id}\">\n{note.strip()}\n</note>"
    out = call_json(instructions, user_input, RELEVANCE_SCHEMA, "relevance_judgment",
                    model=model, reasoning_effort=effort, usage=usage)
    return {"query": query, "patient_id": patient_id, "relevant": bool(out["relevant"]),
            "rationale": out["rationale"], "judge_model": model, "prompt_file": prompt_file}


def judge_all(pairs: List[Tuple[str, int]], cache: JsonlCache, model: str, effort: str,
              workers: int, prompt_file: str = PROMPT_FILE) -> Usage:
    """Judge every (query, patient) pair not already in the cache, in parallel."""
    todo = [(q, p) for q, p in dict.fromkeys(pairs) if pair_key(q, p, prompt_file) not in cache]
    usage = Usage()
    print(f"[judge_retrieval] {len(set(pairs)):,} unique pairs, {len(todo):,} new to judge "
          f"({len(set(pairs)) - len(todo):,} cached)")
    if not todo:
        return usage
    notes = load_notes(p for _, p in todo)
    instructions = load_prompt(prompt_file)
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge_pair, q, p, notes[p], instructions, model, effort, usage, prompt_file): (q, p)
                   for q, p in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="judging"):
            q, p = futures[fut]
            try:
                cache.put(pair_key(q, p, prompt_file), fut.result())
            except Exception as e:          # one bad pair must not kill a 3,000-call run
                failures += 1
                tqdm.write(f"[judge_retrieval] failed on patient {p}: {type(e).__name__}: {e}")
    if failures:
        print(f"[judge_retrieval] {failures} pairs failed; re-run the same command to retry just those")
    return usage


# =========================================================================== metrics
def dcg(rels: List[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def query_metrics(ranked: List[int], labels: Dict[int, bool], gold, pool_relevant: set,
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
            m[f"gold_recall@{c}"] = None if gold is None else float(gold in ranked[:c])
    m[f"any_relevant@{k}"] = float(sum(rels) > 0)
    m["prec_all"] = sum(rels) / len(ranked) if ranked else None      # precision over everything returned
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
                    help="bm25-raw | dense:<model> | results:<results.jsonl> | shown:<results.jsonl> | "
                         "hybrid:<results.jsonl>")
    ap.add_argument("--prompt", default="auto",
                    help=f"relevance prompt in prompts/: 'auto' = {PROMPT_FILE} if the queries have a gold "
                         f"patient_id, else {COHORT_PROMPT_FILE}")
    ap.add_argument("--dense-model", default="medembed-base", help="dense model used by hybrid systems")
    ap.add_argument("--w-dense", type=float, nargs="+", default=[0.5],
                    help="hybrid: weight of dense votes (BM25 = 1.0); several values = several systems")
    ap.add_argument("--rrf-k", type=float, nargs="+", default=[60], help="hybrid: RRF constant(s)")
    ap.add_argument("--depth", type=int, nargs="+", default=[100],
                    help="hybrid: how many results of each list take part in fusion")
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH)
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries (for a test)")
    ap.add_argument("-k", type=int, default=20)
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--effort", default="low", help="judge reasoning effort")
    ap.add_argument("--workers", type=int, default=8, help="parallel judge calls")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    queries = read_jsonl(args.queries)[:args.limit]
    gold = {q["query"]: (int(q["patient_id"]) if q.get("patient_id") is not None else None) for q in queries}
    has_gold = any(g is not None for g in gold.values())
    prompt_file = args.prompt if args.prompt != "auto" else (PROMPT_FILE if has_gold else COHORT_PROMPT_FILE)
    print(f"[judge_retrieval] relevance prompt: {prompt_file}")

    # 1. ranked lists from every system (a hybrid spec expands into one system per setting)
    settings = list(itertools.product(args.w_dense, args.rrf_k, args.depth))
    rankings: Dict[str, Dict[str, List[int]]] = {}
    for s in args.systems:
        if s.startswith("hybrid:"):
            rankings.update(get_hybrid_rankings(Path(s.split(":", 1)[1]), queries, args.k,
                                                args.dense_model, settings))
        else:
            rankings[system_label(s)] = get_rankings(s, queries, args.k)

    # verifier verdicts from runs made with --verify (every screened note, kept or dropped)
    qset = {q["query"] for q in queries}
    verdicts_by_run: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for spec in args.systems:
        if spec.startswith(("results:", "shown:")):
            run = Path(spec.split(":", 1)[1])
            vs = [(r["query"], v) for r in read_jsonl(run) if r["query"] in qset
                  for srch in (r.get("searches") or []) for v in (srch.get("verdicts") or [])]
            if vs:
                verdicts_by_run[run.parent.name] = vs

    # 2. judge the pool: union of every system's top-k and every screened note (cached pairs are free)
    cache = JsonlCache(CACHE_PATH)
    pairs = [(q, p) for ranks in rankings.values() for q, ids in ranks.items() for p in ids]
    pairs += [(q, int(v["patient_id"])) for vs in verdicts_by_run.values() for q, v in vs]
    usage = judge_all(pairs, cache, args.model, args.effort, args.workers, prompt_file)

    def label(q: str, p: int):
        rec = cache.get(pair_key(q, p, prompt_file))
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
        summary[name] = {}
        for key in keys:
            vals = [r[key] for r in rows if r[key] is not None]
            summary[name][key] = round(sum(vals) / len(vals), 4) if vals else None

    # verifier agreement: its keep/drop decision vs the judge's label, on every screened note
    verifier_agreement = {}
    for run_name, vs in verdicts_by_run.items():
        tp = fp = fn = tn = 0
        by_label: Dict[str, List[int]] = {}
        for q, v in vs:
            lab = label(q, int(v["patient_id"]))
            if lab is None:
                continue                                  # judge call failed for this pair
            by_label.setdefault(v["label"], []).append(int(lab))
            if v["kept"]:
                tp, fp = tp + lab, fp + (not lab)
            else:
                fn, tn = fn + lab, tn + (not lab)
        n = tp + fp + fn + tn
        verifier_agreement[run_name] = {
            "n_screened": n, "kept": tp + fp, "dropped": fn + tn,
            "kept_precision": round(tp / (tp + fp), 4) if tp + fp else None,    # kept notes that are relevant
            "kept_recall": round(tp / (tp + fn), 4) if tp + fn else None,       # relevant notes that were kept
            "drop_precision": round(tn / (tn + fn), 4) if tn + fn else None,    # dropped notes that are irrelevant
            "accuracy": round((tp + tn) / n, 4) if n else None,
            "judge_relevant_rate_by_verifier_label": {k: round(sum(x) / len(x), 3) for k, x in by_label.items()},
        }

    # 4. save + print
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = EVAL_DIR / "retrieval" / (f"{stamp}_{args.tag}" if args.tag else stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_query.jsonl", "w") as f:
        for row in per_query:
            f.write(json.dumps(row) + "\n")
    hybrid_settings = ({"dense_model": args.dense_model, "w_dense": args.w_dense, "rrf_k": args.rrf_k,
                        "depth": args.depth} if any(s.startswith("hybrid:") for s in args.systems) else None)
    json.dump({"systems": args.systems, "hybrid_settings": hybrid_settings, "relevance_prompt": prompt_file,
               "queries_file": str(args.queries), "verifier_agreement": verifier_agreement,
               "k": args.k, "n_queries": len(queries), "judge_model": args.model,
               "judge_effort": args.effort, "usage_this_run": usage.as_dict(), "summary": summary},
              open(out_dir / "summary.json", "w"), indent=1)

    cols = [c for c in next(iter(summary.values())) if any(s[c] is not None for s in summary.values())]
    width = max(len(n) for n in summary) + 2
    print("\n" + "system".ljust(width) + "".join(c.rjust(17) for c in cols))
    for name, s in summary.items():
        print(name.ljust(width) + "".join(f"{s[c]:17.3f}" if s[c] is not None else f"{'-':>17}" for c in cols))
    for run_name, a in verifier_agreement.items():
        print(f"\nverifier vs judge ({run_name}): screened {a['n_screened']}, kept {a['kept']}, dropped {a['dropped']}")
        print(f"   kept precision {a['kept_precision']}   kept recall {a['kept_recall']}   "
              f"drop precision {a['drop_precision']}   accuracy {a['accuracy']}")
        print(f"   judge-relevant rate by verifier label: {a['judge_relevant_rate_by_verifier_label']}")
    print(f"\njudge usage this run: {usage}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
