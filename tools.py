"""
Tools and the tool registry.

A Tool is: a name, the JSON schema the LLM sees, a Python handler, and an optional call limit.
The orchestrator looks tools up by name in the ToolRegistry and runs them. Adding a new tool
means writing a handler and registering it: the loop doesn't change.

Handler contract:
    handler(args: dict, state: RunState) -> ToolResult
      - ToolResult.output   is the text the LLM will read
      - ToolResult.artifact is structured data for US (stored in state.artifacts, never shown to the LLM)
      - raise ValueError for bad arguments: the orchestrator turns it into an error message for the LLM
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from bm25_search import BM25Search
from state import RunState


@dataclass
class ToolResult:
    output: str
    artifact: Any = None


@dataclass
class Tool:
    name: str
    schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any], RunState], ToolResult]
    max_calls: Optional[int] = None          # None = unlimited


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name != tool.schema.get("name"):
            raise ValueError(f"tool name {tool.name!r} != schema name {tool.schema.get('name')!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    def remaining(self, name: str, state: RunState) -> Optional[int]:
        """Calls left for this tool in this run (None = unlimited)."""
        tool = self._tools[name]
        if tool.max_calls is None:
            return None
        return max(0, tool.max_calls - state.tool_calls.get(name, 0))

    def has_budget(self, state: RunState) -> bool:
        """True if at least one tool can still be called."""
        return any(self.remaining(n, state) != 0 for n in self._tools)


# =========================================================================== BM25 tool
@dataclass
class SearchCall:
    core_keywords: List[str]
    auxiliary_keywords: List[str]
    ranked: List[Tuple[int, float]]          # ALL patients, (patient_id, score), sorted
    top_k_ids: List[int]                     # the K patients the agent actually got to read


def make_bm25_schema(top_k: int) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "bm25_search",
        "description": (
            f"Keyword (BM25) search over all patients' discharge summaries. Matches exact words after "
            f"stemming; no semantic matching. Returns the top {top_k} notes, each labeled with its patient id."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "core_keywords": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Key clinical entities taken directly from the question "
                                   "(conditions, procedures, drugs, tests). Short and specific.",
                },
                "auxiliary_keywords": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Expansion terms not in the question that are likely to appear in "
                                   "relevant notes: synonyms, abbreviations, related tests, treatments, "
                                   "complications. Weighted lower than core keywords.",
                },
            },
            "required": ["core_keywords", "auxiliary_keywords"],
            "additionalProperties": False,
        },
    }


def format_observation(bm25: BM25Search, call: SearchCall) -> str:
    scores = dict(call.ranked[:len(call.top_k_ids)])
    hits = [pid for pid in call.top_k_ids if scores[pid] > 0]   # drop zero-score (non-matching) notes

    header = (f"Search: core={call.core_keywords} auxiliary={call.auxiliary_keywords}\n"
              f"Retrieved {len(hits)} note(s).")
    if not hits:
        return header + "\nNo notes matched these keywords. Try different keywords, or say the corpus has no evidence."

    blocks = [f'<note patient_id="{pid}">\n{bm25.get_note(pid).strip()}\n</note>' for pid in hits]
    footer = ("Answer using ONLY the notes above. Many may be irrelevant: ignore those. "
              "Cite every claim as [patient <id>] using the patient_id attribute.")
    return "\n\n".join([header, *blocks, footer])


def make_bm25_tool(bm25: BM25Search, top_k: int = 20, aux_weight: float = 0.5,
                   max_calls: Optional[int] = 1) -> Tool:
    def handler(args: Dict[str, Any], state: RunState) -> ToolResult:
        core, aux = args.get("core_keywords"), args.get("auxiliary_keywords", [])
        if not isinstance(core, list) or not core:
            raise ValueError("core_keywords must be a non-empty list of strings")
        if not isinstance(aux, list):
            raise ValueError("auxiliary_keywords must be a list of strings")
        core, aux = [str(k) for k in core], [str(k) for k in aux]
        ranked = bm25.search(core, aux, aux_weight=aux_weight)                    # all patients
        call = SearchCall(core, aux, ranked, [pid for pid, _ in ranked[:top_k]])
        return ToolResult(output=format_observation(bm25, call), artifact=call)

    return Tool(name="bm25_search", schema=make_bm25_schema(top_k), handler=handler, max_calls=max_calls)
