"""
The orchestrator: the runtime that drives a run from start to finish.

It owns the RunState and is the only thing that changes it. Each step:

    1. ask the agent for its next move          agent.step(state, tools, allow_tools)
    2. record the LLM call                      usage, output items, events
    3. if the agent answered  -> finish
       if it called tools     -> dispatch each one through the registry, append the outputs
    4. check limits, save a checkpoint, repeat

What lives HERE (and not in the agent or the prompt): the loop, step and tool limits,
which tool runs for which call, error handling, run status, and persistence.

Main methods:
    _dispatch()   run ONE function call safely and return the text the LLM will read
    run()         the control loop with status transitions

Run:    python orchestrator.py                       all queries in QUERIES_PATH -> results/<run_name>/
        python orchestrator.py --limit 5 --tag trial  quick trial on the first 5 queries
        python orchestrator.py --resume results/<run_name>   continue an interrupted run
        python orchestrator.py --one 0               debug: one query, print everything
Tests:  python test_orchestrator.py     (fake model, no API calls)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from agent import AUX_WEIGHT, TOP_K, Agent, check_grounding
from bm25_search import BM25Search
from state import RunState, RunStatus
from tools import ToolRegistry, make_bm25_tool

MAX_STEPS = 4                      # max LLM calls per run (before the forced final answer)
CHECKPOINT_DIR = Path("runs")
RESULTS_DIR = Path("results")
QUERIES_PATH = Path("data/asclepius/general_temporal_queries/temporal_information_queries_top100.jsonl")


class Orchestrator:
    def __init__(self, agent: Agent, registry: ToolRegistry, max_steps: int = MAX_STEPS,
                 checkpoint_dir: Optional[Path] = CHECKPOINT_DIR, verbose: bool = True):
        self.agent, self.registry = agent, registry
        self.max_steps, self.checkpoint_dir, self.verbose = max_steps, checkpoint_dir, verbose

    # ------------------------------------------------------------------ helpers
    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _checkpoint(self, state: RunState) -> None:
        """Save the full state after each step (runs/<run_id>.json). Cheap insurance:
        if anything crashes mid-run you can open the file and see exactly where."""
        if self.checkpoint_dir is not None:
            state.save(self.checkpoint_dir / f"{state.run_id}.json")

    def _call_agent(self, state: RunState, allow_tools: bool):
        """One agent step + the bookkeeping every LLM call needs. Returns the AgentStep."""
        step = self.agent.step(state, self.registry.schemas(), allow_tools=allow_tools)
        state.record_usage(step.response)
        state.items.extend(step.output_items)       # ALL output items (incl. reasoning) go back next turn
        state.log("llm", allow_tools=allow_tools, n_function_calls=len(step.function_calls),
                  final=step.is_final, reasoning=step.reasoning_summary[:500])
        if step.reasoning_summary:
            self._log(f"[step {state.step}] reasoning: {step.reasoning_summary[:400]}")
        return step

    # ------------------------------------------------------------------ dispatch
    def _dispatch(self, call: Dict[str, Any], state: RunState) -> str:
        """
        Run ONE function call and return the text the LLM will read as its function_call_output.

        `call` is a dict like:
            {"type": "function_call", "call_id": "call_1", "name": "bm25_search",
             "arguments": '{"core_keywords": ["CHF"], "auxiliary_keywords": ["BNP"]}'}
                          ^ a JSON *string*, not a dict

        Rule: this method NEVER raises. Every failure becomes an error message the model reads and can
        react to. The checks run from cheapest to most expensive; each one returns early on failure.
        """
        name = call.get("name")

        # ---- check 1: does the tool exist?
        tool = self.registry.get(name)
        if tool is None:
            state.log("tool_error", name=name, error="unknown tool")
            return f"Error: unknown tool {name!r}. Available tools: {self.registry.names()}."

        # ---- check 2: is its call limit used up?
        if self.registry.remaining(name, state) == 0:
            state.log("tool_refused", name=name, reason="call limit")
            return f"Error: the call limit for {name} is used up. Answer with the information you already have."

        # ---- check 3: are the arguments valid JSON?
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError as e:
            # Note: we return BEFORE counting the call, so bad JSON doesn't use up the limit.
            state.log("tool_error", name=name, error=f"bad json: {e}")
            return f"Error: arguments are not valid JSON ({e}). Call {name} again with valid JSON."

        # ---- run the tool: two kinds of failure, two messages
        try:
            result = tool.handler(args, state)
        except ValueError as e:                 # the handler says the arguments are wrong
            state.log("tool_error", name=name, args=args, error=str(e))
            return f"Error: invalid arguments for {name}: {e}"
        except Exception as e:                  # the handler itself crashed (a bug, or BM25 failed)
            state.log("tool_error", name=name, args=args, error=repr(e))
            return f"Error: {name} failed ({type(e).__name__}). Answer with the information you already have."

        # ---- success: record it in the state
        state.tool_calls[name] = state.tool_calls.get(name, 0) + 1
        if result.artifact is not None:
            state.artifacts.setdefault(name, []).append(result.artifact)

        state.log("tool", name=name, args=args)
        self._log(f"[step {state.step}] {name}({args})")
        return result.output                    # the text the model will read

    # ------------------------------------------------------------------ the loop
    def run(self, query: str) -> RunState:
        """
        Drive one run from start to finish and return the final RunState.
        The status starts as RUNNING and changes exactly once, to ANSWERED, BUDGET_EXHAUSTED or ERROR.
        """
        state = RunState(query=query)
        state.items = self.agent.initial_items(query)
        self._log(f"\nRUN {state.run_id}  QUERY: {query}")

        try:
            while not state.done:                         # done = status is no longer RUNNING

                # ---- out of steps: one last call with tools OFF, so the model has to answer
                if state.step >= self.max_steps:
                    state.step += 1
                    state.log("forced_answer", reason="max_steps")
                    step = self._call_agent(state, allow_tools=False)
                    state.answer = step.text or None
                    state.status = RunStatus.ANSWERED if state.answer else RunStatus.BUDGET_EXHAUSTED
                    break

                # ---- a normal step: ask the agent what to do next
                state.step += 1
                allow_tools = self.registry.has_budget(state)       # tools switch off once every limit is used
                step = self._call_agent(state, allow_tools=allow_tools)  # LLM call + usage + items + log

                if step.is_final:
                    # ---- the agent answered: we're done (state.done becomes True, the loop stops)
                    state.answer = step.text
                    state.status = RunStatus.ANSWERED
                else:
                    # ---- the agent called tool(s): run each one and send its result back
                    for call in step.function_calls:
                        output = self._dispatch(call, state)
                        # every function_call gets exactly one output, linked by the same call_id
                        state.items.append({"type": "function_call_output",
                                            "call_id": call["call_id"],
                                            "output": output})

                self._checkpoint(state)                   # save after every step

        except Exception as e:
            # ---- something crashed (e.g. the API is down). Don't crash the caller: record it and return.
            # This also catches bugs in our own code, so always check state.error on ERROR runs.
            state.status = RunStatus.ERROR
            state.error = f"{type(e).__name__}: {e}"
            state.log("error", error=state.error)
            self._log(f"[step {state.step}] ERROR {state.error}")

        state.finished_at = time.time()
        self._checkpoint(state)                           # final save
        return state


# =========================================================================== main
def build(top_k: int = TOP_K, max_searches: int = 1) -> Orchestrator:
    bm25 = BM25Search.load_or_build()
    registry = ToolRegistry()
    registry.register(make_bm25_tool(bm25, top_k=top_k, aux_weight=AUX_WEIGHT, max_calls=max_searches))
    return Orchestrator(Agent(), registry)


def _git_commit() -> Optional[str]:
    """Short git commit hash of this folder, if it's a git repo (helps tie a run to a code version)."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _run_config(args, orch: Orchestrator, queries_path: Path, n_queries: int) -> Dict[str, Any]:
    """Everything needed to know later what this run was: model, settings, prompt, data."""
    import hashlib
    import agent as A
    return {
        "run_name": args.run_name,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag,
        "git_commit": _git_commit(),
        "model": orch.agent.model,
        "reasoning_effort": orch.agent.reasoning_effort,
        "top_k": TOP_K,
        "aux_weight": AUX_WEIGHT,
        "max_steps": orch.max_steps,
        "max_searches": args.max_searches,
        "queries_file": str(queries_path),
        "n_queries": n_queries,
        "system_prompt_file": A.SYSTEM_PROMPT_FILE,
        "system_prompt_sha256": hashlib.sha256(A.SYSTEM_PROMPT.encode()).hexdigest()[:12],
        "system_prompt": A.SYSTEM_PROMPT,          # full text, so you can diff prompts between runs
        "user_prompt_template": A.USER_PROMPT_TEMPLATE,
    }


def _result_record(item: Dict[str, Any], state: RunState) -> Dict[str, Any]:
    """One line of the results file: the query, the model's answer, and the top-k patients shown."""
    return {
        "query": item["query"],
        "gold_patient_id": item["patient_id"],
        "answer": state.answer,
        "top_k_patient_ids": [pid for s in state.searches for pid in s.top_k_ids],   # in rank order
        "searches": [{"core_keywords": s.core_keywords, "auxiliary_keywords": s.auxiliary_keywords}
                     for s in state.searches],
        "status": state.status.value,
        "error": state.error,
        "run_id": state.run_id,                   # -> traces/<run_id>.json has the full conversation
        "n_llm_calls": state.n_llm_calls,
        "input_tokens": state.input_tokens,
        "output_tokens": state.output_tokens,
        "reasoning_tokens": state.reasoning_tokens,
        "seconds": round((state.finished_at or time.time()) - state.started_at, 1),
    }


def run_one_verbose(orch: Orchestrator, item: Dict[str, Any]) -> None:
    """Debug mode: run a single query and print everything (the old behaviour of this script)."""
    state = orch.run(item["query"])
    print("\n" + "=" * 100)
    print(f"RUN {state.run_id}  status={state.status.value}  steps={state.step}  error={state.error}")
    print("QUERY:  ", item["query"])
    for i, s in enumerate(state.searches, 1):
        print(f"SEARCH {i}: core={s.core_keywords}\n          aux={s.auxiliary_keywords}")
        gold_rank = next((r for r, (p, _) in enumerate(s.ranked, 1) if p == item["patient_id"]), None)
        print(f"          gold patient {item['patient_id']} rank = {gold_rank}  (in top-{TOP_K}: {item['patient_id'] in s.top_k_ids})")
    print(f"LLM calls: {state.n_llm_calls} | tokens in={state.input_tokens:,} out={state.output_tokens:,} "
          f"(reasoning={state.reasoning_tokens:,})")
    print("EVENTS:", [e.kind for e in state.events])
    print("\nANSWER:\n", state.answer)
    print("\nGROUNDING:", check_grounding(state))


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    ap = argparse.ArgumentParser(description="Run the agent over a queries file and save the answers.")
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH, help="queries .jsonl (query, patient_id, answer)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries (for a quick trial)")
    ap.add_argument("--tag", default="", help="short label for this run, e.g. 'baseline' or 'aux-weight-0.3'")
    ap.add_argument("--max-searches", type=int, default=1)
    ap.add_argument("--resume", type=Path, default=None,
                    help="an existing results/<run_name> folder: skip queries already done and append")
    ap.add_argument("--one", type=int, default=None, help="debug: run only query #N and print everything")
    args = ap.parse_args()

    queries = [json.loads(l) for l in open(args.queries) if l.strip()]
    if args.limit:
        queries = queries[:args.limit]
    orch = build(max_searches=args.max_searches)

    if args.one is not None:                                   # debug mode
        orch.checkpoint_dir = CHECKPOINT_DIR
        run_one_verbose(orch, queries[args.one])
        raise SystemExit

    # ---- one folder per run:  results/<run_name>/{<run_name>.jsonl, config.json, traces/}
    if args.resume:
        run_dir = args.resume
        args.run_name = run_dir.name
    else:
        args.run_name = "_".join(x for x in [time.strftime("%Y%m%d-%H%M%S"), orch.agent.model,
                                             args.tag.replace(" ", "-")] if x)
        run_dir = RESULTS_DIR / args.run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps(_run_config(args, orch, args.queries, len(queries)), indent=1))
    results_path = run_dir / f"{args.run_name}.jsonl"
    orch.checkpoint_dir = run_dir / "traces"                   # full per-query state, for debugging
    orch.verbose = False                                       # keep the console clean for tqdm

    done = set()
    if results_path.exists():
        done = {json.loads(l)["query"] for l in open(results_path) if l.strip()}
    todo = [q for q in queries if q["query"] not in done]
    print(f"run: {args.run_name}\n{len(done)} already done, {len(todo)} to run -> {results_path}")

    n_err = 0
    with open(results_path, "a") as f:
        for item in tqdm(todo, desc="queries", unit="q"):
            state = orch.run(item["query"])                    # never raises: errors end up in state
            n_err += state.status is RunStatus.ERROR
            f.write(json.dumps(_result_record(item, state), ensure_ascii=False) + "\n")
            f.flush()                                          # each line is saved as soon as it's done
    print(f"done. {len(todo) - n_err} ok, {n_err} errors (status='error' lines can be re-run with --resume "
          f"after deleting them). Results: {results_path}")
