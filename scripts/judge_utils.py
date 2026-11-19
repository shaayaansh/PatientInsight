"""
Shared plumbing for the two LLM judges (judge_retrieval.py, judge_answers.py).

    call_json()     one Responses API call that must return JSON matching a schema (with retries)
    Usage           thread-safe token counter, printed at the end of a run
    JsonlCache      append-only cache on disk: a judgment is never paid for twice
    parse_citations the same [patient 123] parsing as agent.check_grounding
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import openai
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]      # repo root (this file lives in scripts/)
PROMPTS_DIR = ROOT / "prompts"
DATA_PATH = ROOT / "data/asclepius/asclepius.parquet"
QUERY_DIR = ROOT / "data/asclepius/general_temporal_queries"
QUERIES_PATH = QUERY_DIR / "temporal_information_queries_top100.jsonl"
RUBRIC_PATH = QUERY_DIR / "evaluation_rubric.jsonl"

JUDGE_MODEL = "gpt-5.6"     # deliberately a different (stronger) model than the agent

client = OpenAI(api_key=(ROOT / "openai_secret.txt").read_text().strip(), max_retries=0)  # we retry ourselves

# errors worth retrying: rate limits, timeouts, network, 5xx. Anything else (e.g. 400) is a bug -> raise.
RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
             openai.InternalServerError)


# --------------------------------------------------------------------------- small helpers
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text().strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_notes(patient_ids: Optional[Iterable[int]] = None) -> Dict[int, str]:
    """patient_id -> note text, read straight from the parquet (no search index needed)."""
    import pandas as pd
    df = pd.read_parquet(DATA_PATH, columns=["patient_id", "note"])
    if patient_ids is not None:
        df = df[df["patient_id"].isin(set(int(p) for p in patient_ids))]
    return {int(p): n for p, n in zip(df["patient_id"], df["note"])}


# same pattern as agent.CITATION_GROUP_RE: any [...] group that contains the word patient(s)
CITATION_GROUP_RE = re.compile(r"\[([^\[\]]*?\bpatients?\b[^\[\]]*?)\]", re.IGNORECASE)


def parse_citations(answer: Optional[str]) -> Set[int]:
    cited: Set[int] = set()
    for group in CITATION_GROUP_RE.findall(answer or ""):
        cited.update(int(n) for n in re.findall(r"\d+", group))
    return cited


# --------------------------------------------------------------------------- usage
class Usage:
    def __init__(self):
        self._lock = threading.Lock()
        self.calls = self.input_tokens = self.output_tokens = self.reasoning_tokens = 0

    def add(self, response) -> None:
        u = getattr(response, "usage", None)
        with self._lock:
            self.calls += 1
            if u is None:
                return
            self.input_tokens += u.input_tokens or 0
            self.output_tokens += u.output_tokens or 0
            details = getattr(u, "output_tokens_details", None)
            self.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0

    def as_dict(self) -> Dict[str, int]:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "reasoning_tokens": self.reasoning_tokens}

    def __str__(self) -> str:
        return (f"{self.calls} calls, {self.input_tokens:,} input / {self.output_tokens:,} output tokens "
                f"({self.reasoning_tokens:,} of them reasoning)")


# --------------------------------------------------------------------------- the LLM call
def call_json(instructions: str, user_input: str, schema: Dict[str, Any], schema_name: str,
              model: str = JUDGE_MODEL, reasoning_effort: str = "low",
              usage: Optional[Usage] = None, max_tries: int = 6) -> Dict[str, Any]:
    """
    One Responses API call whose output is forced to be JSON matching `schema` (structured outputs,
    strict mode). Retries transient errors with exponential backoff + jitter.
    """
    for attempt in range(max_tries):
        try:
            resp = client.responses.create(
                model=model,
                instructions=instructions,
                input=user_input,
                reasoning={"effort": reasoning_effort},
                text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
            )
            if usage is not None:
                usage.add(resp)
            return json.loads(resp.output_text)
        except RETRYABLE:
            if attempt == max_tries - 1:
                raise
            time.sleep(min(60, 2 ** attempt) + random.random())
        except json.JSONDecodeError:   # truncated / refused output: try again
            if attempt == max_tries - 1:
                raise
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- cache
class JsonlCache:
    """
    Append-only key -> record store. Each line is {"key": ..., **record}. Loaded fully at start;
    every new record is appended (and flushed) immediately, so a crash loses at most one judgment.
    Thread-safe for the parallel judges.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.data: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            for rec in read_jsonl(self.path):
                self.data[rec["key"]] = rec

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def put(self, key: str, record: Dict[str, Any]) -> None:
        rec = {"key": key, **record}
        with self._lock:
            self.data[key] = rec
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
