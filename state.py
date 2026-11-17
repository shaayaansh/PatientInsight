"""
RunState: everything about ONE run of the system, in one place.

The orchestrator owns this object and is the only thing that changes it. The agent reads it
(to build the next LLM call) but never writes to it. That separation is the whole point:
  - you can save a run (checkpoint) at any step, because the state is plain data
  - you can inspect / replay a run after the fact
  - limits (steps, tool calls) are counted in one place
"""
from __future__ import annotations

import dataclasses
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunStatus(str, Enum):
    RUNNING = "running"
    ANSWERED = "answered"                  # the agent produced a final answer
    BUDGET_EXHAUSTED = "budget_exhausted"  # hit max_steps and could not produce an answer
    ERROR = "error"                        # something crashed (API error, bug)


@dataclass
class Event:
    """One line in the run's history log (llm call, tool call, error, ...)."""
    step: int
    kind: str
    data: Dict[str, Any]
    t: float = field(default_factory=time.time)


@dataclass
class RunState:
    query: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: RunStatus = RunStatus.RUNNING
    step: int = 0                                         # number of LLM calls started so far

    items: List[Dict[str, Any]] = field(default_factory=list)          # conversation, as JSON-able dicts
    artifacts: Dict[str, List[Any]] = field(default_factory=dict)      # tool name -> its structured results
    tool_calls: Dict[str, int] = field(default_factory=dict)           # tool name -> times executed
    events: List[Event] = field(default_factory=list)

    answer: Optional[str] = None
    error: Optional[str] = None

    n_llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0                                # includes reasoning tokens
    reasoning_tokens: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    # ------------------------------------------------------------------ helpers
    @property
    def done(self) -> bool:
        return self.status is not RunStatus.RUNNING

    @property
    def searches(self) -> List[Any]:
        """BM25 SearchCall artifacts (kept under this name so check_grounding works unchanged)."""
        return self.artifacts.get("bm25_search", [])

    def log(self, kind: str, **data) -> None:
        self.events.append(Event(self.step, kind, data))

    def record_usage(self, response) -> None:
        """Add one Responses API call's token usage to the running totals."""
        self.n_llm_calls += 1
        u = getattr(response, "usage", None)
        if u is None:
            return
        self.input_tokens += u.input_tokens or 0
        self.output_tokens += u.output_tokens or 0
        details = getattr(u, "output_tokens_details", None)
        self.reasoning_tokens += (getattr(details, "reasoning_tokens", 0) or 0) if details else 0

    # ------------------------------------------------------------------ persistence
    def to_dict(self, max_ranked: int = 1000) -> Dict[str, Any]:
        """JSON-able snapshot. Big BM25 rankings are cut to `max_ranked` entries."""
        def artifact_to_dict(a):
            if dataclasses.is_dataclass(a):
                d = dataclasses.asdict(a)
                if "ranked" in d:
                    d["ranked"] = d["ranked"][:max_ranked]
                return d
            return a
        return {
            "run_id": self.run_id, "query": self.query, "status": self.status.value, "step": self.step,
            "answer": self.answer, "error": self.error,
            "items": self.items,
            "artifacts": {k: [artifact_to_dict(a) for a in v] for k, v in self.artifacts.items()},
            "tool_calls": self.tool_calls,
            "events": [dataclasses.asdict(e) for e in self.events],
            "n_llm_calls": self.n_llm_calls, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "reasoning_tokens": self.reasoning_tokens,
            "started_at": self.started_at, "finished_at": self.finished_at,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1, default=str))
        tmp.replace(path)            # write-then-rename, so a crash never leaves a half-written file
