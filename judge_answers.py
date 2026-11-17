"""
Answer judge: grade every answer of an orchestrator run against evaluation_rubric.jsonl.

One gpt-5.6 call per answer. The judge sees the question, the answer, the full notes of the patients
the answer cites, and the applicable rubric items, and says for each item whether it HOLDS:
    positive item  + holds  -> points earned        (e.g. G1 +8 "every claim is cited and supported")
    negative item  + holds  -> penalty applied      (e.g. G2 -10 "states a time the note doesn't give")

Which items apply:
    general (G1-G10) and query_specific (S*)   always
    gold_patient (S* with applies_if_cited_patient)   only if the answer cites that patient

Score per answer:
    raw   = sum(points of items that hold)                 (penalties are negative)
    max   = sum(positive points of applicable items)
    score = clip(raw / max, 0, 1)
A run with no answer (error / budget exhausted) gets score 0 without calling the judge.

Outputs go next to the run being judged, so results and grades stay together:
    results/<run>/judge_<model>/judgments.jsonl   one line per query (all item verdicts + justifications)
    results/<run>/judge_<model>/summary.json      mean score, per-item hit rates, per-category points
Resumable: queries already in judgments.jsonl are skipped. Re-run the same command after a crash.

Usage (run from the project root):
    python judge_answers.py results/<run>/<run>.jsonl --limit 3      # quick test, look at the justifications
    python judge_answers.py results/<run>/<run>.jsonl
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from judge_utils import (JUDGE_MODEL, RUBRIC_PATH, Usage, call_json, load_notes, load_prompt,
                         parse_citations, read_jsonl)

PROMPT_FILE = "judge_answer.md"


# =========================================================================== building the judge input
def applicable_items(rubric: List[Dict[str, Any]], cited: set) -> List[Dict[str, Any]]:
    return [it for it in rubric
            if it.get("applies_if_cited_patient") is None or int(it["applies_if_cited_patient"]) in cited]


def verdict_schema(item_ids: List[str]) -> Dict[str, Any]:
    """Strict JSON schema whose `id` enum is exactly this answer's applicable items."""
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": item_ids},
                        "justification": {"type": "string"},
                        "holds": {"type": "boolean"},
                    },
                    "required": ["id", "justification", "holds"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }


def build_input(query: str, answer: str, cited_notes: Dict[int, str], fabricated: List[int],
                items: List[Dict[str, Any]]) -> str:
    parts = [f"## Question\n{query}", f"## Assistant's answer\n{answer.strip()}"]
    if cited_notes:
        notes = "\n\n".join(f'<note patient_id="{pid}">\n{note.strip()}\n</note>'
                            for pid, note in sorted(cited_notes.items()))
        parts.append(f"## Notes of the cited patients\n{notes}")
    else:
        parts.append("## Notes of the cited patients\n(The answer cites no retrieved patient.)")
    if fabricated:
        parts.append("## Cited but never retrieved (fabricated citations)\n"
                     + ", ".join(f"patient {p}" for p in fabricated))
    rubric = "\n".join(f"- {it['id']} ({it['points']:+d} points, {it['category']}): {it['criterion']}"
                       for it in items)
    parts.append(f"## Rubric\n{rubric}")
    return "\n\n".join(parts)


# =========================================================================== scoring
def score(items: List[Dict[str, Any]], verdicts: Dict[str, bool]) -> Dict[str, Any]:
    raw = sum(it["points"] for it in items if verdicts.get(it["id"]))
    max_points = sum(it["points"] for it in items if it["points"] > 0)
    return {"raw": raw, "max": max_points,
            "score": max(0.0, min(1.0, raw / max_points)) if max_points else 0.0}


# =========================================================================== judging one answer
def judge_one(rec: Dict[str, Any], rubric: List[Dict[str, Any]], instructions: str, model: str,
              effort: str, usage: Usage, notes_lock: threading.Lock, notes: Dict[int, str]) -> Dict[str, Any]:
    query, answer = rec["query"], rec.get("answer") or ""
    shown = set(int(p) for p in rec.get("top_k_patient_ids", []))
    cited = parse_citations(answer)
    real, fabricated = sorted(cited & shown), sorted(cited - shown)
    base = {"query": query, "gold_patient_id": rec.get("gold_patient_id"), "status": rec.get("status"),
            "cited": sorted(cited), "fabricated_citations": fabricated,
            "gold_cited": rec.get("gold_patient_id") in cited}

    if not answer.strip():                      # nothing to grade
        return {**base, "raw": 0, "max": None, "score": 0.0, "items": [], "note": "no answer"}

    # gold-patient items only apply when the answer cites a REAL (retrieved) gold patient
    items = applicable_items(rubric, set(real))
    with notes_lock:
        cited_notes = {p: notes[p] for p in real if p in notes}
    user_input = build_input(query, answer, cited_notes, fabricated, items)
    ids = [it["id"] for it in items]

    out = call_json(instructions, user_input, verdict_schema(ids), "rubric_verdicts",
                    model=model, reasoning_effort=effort, usage=usage)
    verdicts = {v["id"]: v for v in out["verdicts"]}
    missing = [i for i in ids if i not in verdicts]
    if missing:                                 # schema can't force "every id exactly once": retry once
        out = call_json(instructions, user_input + f"\n\nReturn a verdict for EVERY id, including {missing}.",
                        verdict_schema(ids), "rubric_verdicts", model=model, reasoning_effort=effort, usage=usage)
        verdicts = {v["id"]: v for v in out["verdicts"]}
        missing = [i for i in ids if i not in verdicts]     # still missing -> treated as not holding

    holds = {i: bool(verdicts[i]["holds"]) for i in ids if i in verdicts}
    item_rows = [{"id": it["id"], "type": it["type"], "category": it["category"], "points": it["points"],
                  "holds": holds.get(it["id"]),
                  "justification": verdicts.get(it["id"], {}).get("justification")}
                 for it in items]
    return {**base, **score(items, holds), "items": item_rows, "missing_verdicts": missing,
            "judge_model": model}


# =========================================================================== summary
def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    graded = [r for r in rows if r["items"]]
    item_hits: Dict[str, List[int]] = defaultdict(list)       # general items: how often each holds
    cat_points: Dict[str, float] = defaultdict(float)         # points earned/lost per category
    for r in graded:
        for it in r["items"]:
            if it["holds"] is None:
                continue
            if it["type"] == "general":
                item_hits[it["id"]].append(int(it["holds"]))
            if it["holds"]:
                cat_points[it["category"]] += it["points"]
    return {
        "n_queries": n,
        "n_graded": len(graded),
        "mean_score": round(sum(r["score"] for r in rows) / n, 4) if n else None,
        "mean_raw": round(sum(r["raw"] for r in graded) / len(graded), 2) if graded else None,
        "gold_cited_rate": round(sum(r["gold_cited"] for r in rows) / n, 4) if n else None,
        "fabricated_citation_rate": round(sum(bool(r["fabricated_citations"]) for r in rows) / n, 4) if n else None,
        # e.g. "G2": 0.31 -> the -10 temporal-precision penalty fired on 31% of answers
        "general_item_hold_rate": {k: round(sum(v) / len(v), 3)
                                   for k, v in sorted(item_hits.items(), key=lambda kv: int(kv[0][1:]))},
        "mean_points_per_category": {k: round(v / len(graded), 2) for k, v in sorted(cat_points.items())}
        if graded else {},
    }


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description="Grade an orchestrator run against the evaluation rubric")
    ap.add_argument("results", type=Path, help="results/<run>/<run>.jsonl")
    ap.add_argument("--rubric", type=Path, default=RUBRIC_PATH)
    ap.add_argument("--limit", type=int, default=None, help="only the first N answers (for a test)")
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--effort", default="medium", help="judge reasoning effort")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    records = read_jsonl(args.results)[:args.limit]
    rubrics = {r["query"]: r["eval_rubric"] for r in read_jsonl(args.rubric)}
    no_rubric = [r["query"] for r in records if r["query"] not in rubrics]
    if no_rubric:
        raise SystemExit(f"{len(no_rubric)} queries have no rubric, e.g. {no_rubric[0]!r}")

    out_dir = args.results.parent / f"judge_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "judgments.jsonl"
    done = {r["query"]: r for r in read_jsonl(out_path)} if out_path.exists() else {}
    todo = [r for r in records if r["query"] not in done]
    print(f"[judge_answers] {len(records)} answers, {len(done)} already judged, {len(todo)} to go")

    usage, lock, write_lock = Usage(), threading.Lock(), threading.Lock()
    instructions = load_prompt(PROMPT_FILE)
    notes = load_notes({int(p) for r in todo for p in parse_citations(r.get("answer"))}) if todo else {}
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(out_path, "a") as f:
        futures = {pool.submit(judge_one, r, rubrics[r["query"]], instructions, args.model, args.effort,
                               usage, lock, notes): r for r in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="judging"):
            try:
                row = fut.result()
            except Exception as e:
                failures += 1
                tqdm.write(f"[judge_answers] failed: {futures[fut]['query'][:60]!r}: {type(e).__name__}: {e}")
                continue
            with write_lock:
                f.write(json.dumps(row) + "\n"); f.flush()
            done[row["query"]] = row

    rows = [done[r["query"]] for r in records if r["query"] in done]
    summary = summarize(rows)
    summary.update({"results_file": str(args.results), "judge_model": args.model, "judge_effort": args.effort,
                    "usage_this_run": usage.as_dict(), "failures_this_run": failures})
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)

    print(f"\nmean score  {summary['mean_score']}   (mean raw points {summary['mean_raw']})")
    print(f"gold cited  {summary['gold_cited_rate']}   fabricated citations {summary['fabricated_citation_rate']}")
    print("general items, fraction of answers where each holds:")
    for k, v in summary["general_item_hold_rate"].items():
        print(f"   {k:<4}{v:6.2f}")
    print(f"judge usage this run: {usage}")
    if failures:
        print(f"{failures} answers failed; re-run the same command to retry them")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
