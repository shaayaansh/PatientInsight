"""
The agent: the "brain". Given the current RunState, make ONE LLM call and report what the
model wants to do. It doesn't loop, execute tools, count budgets, or change state: that's the
orchestrator's job (orchestrator.py).

    orchestrator ── state ──▶ agent.step() ──▶ AgentStep(function_calls | final text)
         ▲                                              │
         └──────── runs tools, updates state ◀──────────┘

"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from state import RunState

# --------------------------------------------------------------------------- config
MODEL = "gpt-6-luna"
TOP_K = 20                  # how many patients the agent is allowed to read
AUX_WEIGHT = 0.5            # weight of auxiliary keywords in the BM25 score
REASONING_EFFORT = "low"    # "none" | "low" | "medium" | "high"  (reasoning tokens are billed as output)
REASONING_SUMMARY = "auto"  # readable reasoning summary for logging. None to disable.

client = OpenAI(api_key=(Path(__file__).resolve().parent / "openai_secret.txt").read_text().strip())

# --------------------------------------------------------------------------- prompts
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_FILE = "system.md"
USER_PROMPT_TEMPLATE = "Question: {query}"


def load_prompt(name: str, **kwargs) -> str:
    """Read prompts/<name> and fill {placeholders} with str.format (a literal { or } must be doubled)."""
    return (PROMPTS_DIR / name).read_text().strip().format(**kwargs)


SYSTEM_PROMPT = load_prompt(SYSTEM_PROMPT_FILE, top_k=TOP_K)


# --------------------------------------------------------------------------- agent
def to_dict(item: Any) -> Dict[str, Any]:
    """SDK output item -> plain JSON-able dict (so RunState can be saved as JSON and sent back as input)."""
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)
    if hasattr(item, "__dict__"):                       # e.g. SimpleNamespace in tests
        return {k: ([to_dict(x) for x in v] if isinstance(v, list) else
                    to_dict(v) if hasattr(v, "__dict__") else v)
                for k, v in vars(item).items() if v is not None}
    return item


@dataclass
class AgentStep:
    response: Any                         # raw Response (for usage, status)
    output_items: List[Dict[str, Any]]    # all output items as dicts: append these to state.items
    function_calls: List[Dict[str, Any]]  # the function_call items among them (empty = final answer)
    text: str                             # response.output_text (the answer, if no function calls)
    reasoning_summary: str                # readable summary of the model's reasoning, if any

    @property
    def is_final(self) -> bool:
        return not self.function_calls


class Agent:
    def __init__(self, model: str = MODEL, reasoning_effort: str = REASONING_EFFORT,
                 reasoning_summary: str = REASONING_SUMMARY):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary

    def initial_items(self, query: str) -> List[Dict[str, Any]]:
        """The conversation at the start of a run (the system prompt goes in `instructions`)."""
        return [{"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)}]

    def step(self, state: RunState, tools: List[Dict[str, Any]], allow_tools: bool) -> AgentStep:
        """One LLM call on the current conversation. Reads state, never changes it.
        allow_tools=False -> tool_choice="none": the model sees the tools but must answer."""
        kwargs: Dict[str, Any] = {"model": self.model, "instructions": SYSTEM_PROMPT,
                                  "input": state.items}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto" if allow_tools else "none"
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
            if self.reasoning_summary:
                kwargs["reasoning"]["summary"] = self.reasoning_summary

        response = client.responses.create(**kwargs)

        output_items = [to_dict(it) for it in response.output]
        calls = [it for it in output_items if it.get("type") == "function_call"]
        summary = " ".join(s.get("text", "") for it in output_items if it.get("type") == "reasoning"
                           for s in it.get("summary", []) or [])
        return AgentStep(response, output_items, calls, response.output_text or "", summary)


# --------------------------------------------------------------------------- grounding check
# The prompt asks for "[patient 12345]", but models drift: "[patients 123, 456]", "[Patient 123; 456]".
# So: find every [...] group that mentions "patient", then take every number inside it.
CITATION_GROUP_RE = re.compile(r"\[([^\[\]]*?\bpatients?\b[^\[\]]*?)\]", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+")


def _shown_ids(result) -> List[int]:
    """Patients whose note text was actually in the context (zero-score notes are not shown)."""
    shown: List[int] = []
    for s in result.searches:
        scores = dict(s.ranked[:len(s.top_k_ids)])
        shown += [p for p in s.top_k_ids if scores.get(p, 0) > 0 and p not in shown]
    return shown


def check_grounding(result) -> Dict[str, Any]:
    """Cheap grounding check on result.answer against result.searches (works on RunState).
    Only checks WHO is cited, not whether the claim is in that note (that needs an LLM judge)."""
    answer = result.answer or ""
    shown = _shown_ids(result)
    shown_set = set(shown)
    mentions: List[int] = []
    for group in CITATION_GROUP_RE.findall(answer):
        mentions += [int(n) for n in NUMBER_RE.findall(group)]
    cited = list(dict.fromkeys(mentions))
    hallucinated = [p for p in cited if p not in shown_set]
    return {
        "has_answer": bool(answer.strip()),
        "has_citations": bool(cited),
        "n_citation_mentions": len(mentions),
        "cited": cited,
        "hallucinated": hallucinated,
        "n_shown": len(shown),
        "frac_shown_cited": round(len(set(cited) & shown_set) / len(shown), 3) if shown else 0.0,
        "grounded": bool(cited) and not hallucinated,
    }
