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

import dataclasses
from dataclasses import dataclass, field
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


# =========================================================================== search tools
# Two backends, one tool factory:
#   "bm25"    tool `bm25_search`: keywords only                               (the baseline)
#   "hybrid"  tool `search`:      keywords (BM25) + a natural-language query (dense), fused with RRF
# Either can be wrapped with a Verifier: retrieve -> verify -> only kept notes (+ reasons) reach the agent.
SEARCH_TOOL_NAMES = ("bm25_search", "search")    # RunState.searches collects artifacts from these


@dataclass
class SearchCall:
    core_keywords: List[str]
    auxiliary_keywords: List[str]
    ranked: List[Tuple[int, float]]          # ranked patients (BM25: all; hybrid: every patient that got a vote)
    top_k_ids: List[int]                     # the top-k retrieved (before verification)
    query: str = ""                          # dense / semantic query (hybrid only)
    backend: str = "bm25"
    shown_ids: List[int] = field(default_factory=list)       # notes the agent actually read
    verdicts: Optional[List[Dict[str, Any]]] = None          # one per retrieved note, if verified


KEYWORD_PROPS = {
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
}


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
            "properties": dict(KEYWORD_PROPS),
            "required": ["core_keywords", "auxiliary_keywords"],
            "additionalProperties": False,
        },
    }


def make_hybrid_schema(top_k: int) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "search",
        "description": (
            "Hybrid search over all patients' discharge summaries. Combines keyword search (BM25, driven by "
            "core_keywords and auxiliary_keywords) with semantic search (driven by query) and merges the two "
            f"rankings. Returns the top {top_k} notes, each labeled with its patient id."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                **KEYWORD_PROPS,
                "query": {
                    "type": "string",
                    "description": "The information need as one or two plain clinical sentences, for semantic "
                                   "search (e.g. 'Diabetic ketoacidosis treated with an insulin infusion and later "
                                   "switched to subcutaneous insulin'). Not a keyword list.",
                },
            },
            "required": ["core_keywords", "auxiliary_keywords", "query"],
            "additionalProperties": False,
        },
    }


def format_observation(searcher: Any, call: SearchCall) -> str:
    """The text the agent reads: a header, one <note> block per shown note, a footer."""
    lines = [f"Search: core={call.core_keywords} auxiliary={call.auxiliary_keywords}"]
    if call.query:
        lines.append(f"Semantic query: {call.query!r}")
    n_retrieved = len(call.top_k_ids) if call.backend != "bm25" else len(
        [p for p, s in call.ranked[:len(call.top_k_ids)] if s > 0])
    if call.verdicts is not None:
        lines.append(f"Retrieved {n_retrieved} note(s); {len(call.shown_ids)} passed relevance screening.")
    else:
        lines.append(f"Retrieved {len(call.shown_ids)} note(s).")
    header = "\n".join(lines)

    if not call.shown_ids:
        what = "passed relevance screening" if call.verdicts is not None else "matched this search"
        return (header + f"\nNo notes {what}. Try different search terms if you can still search; "
                         "otherwise say the retrieved notes contain no evidence for the question.")

    reasons = {v["patient_id"]: v for v in (call.verdicts or [])}
    blocks = []
    for pid in call.shown_ids:
        v = reasons.get(pid)
        if v is None:
            blocks.append(f'<note patient_id="{pid}">\n{searcher.get_note(pid).strip()}\n</note>')
        else:
            blocks.append(f'<note patient_id="{pid}" screening="{v["label"]}">\n'
                          f'Screening summary (a pointer, not evidence): {v["reason"]}\n\n'
                          f'{searcher.get_note(pid).strip()}\n</note>')
    footer = "Answer using ONLY the notes above."
    if call.verdicts is not None:
        footer += (" Each note starts with a short screening summary that points to the relevant part; verify "
                   "every claim against the note text itself, never cite the summary as evidence.")
    else:
        footer += " Many may be irrelevant: ignore those."
    footer += " Cite every claim as [patient <id>] using the patient_id attribute."
    return "\n\n".join([header, *blocks, footer])


def _check_keywords(args: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    core, aux = args.get("core_keywords"), args.get("auxiliary_keywords", [])
    if not isinstance(core, list) or not core:
        raise ValueError("core_keywords must be a non-empty list of strings")
    if not isinstance(aux, list):
        raise ValueError("auxiliary_keywords must be a list of strings")
    return [str(k) for k in core], [str(k) for k in aux]


def make_search_tool(searcher: Any, backend: str = "hybrid", top_k: int = 20, aux_weight: float = 0.5,
                     max_calls: Optional[int] = 1, verifier: Any = None) -> Tool:
    """
    searcher  BM25Search (backend="bm25") or HybridSearch (backend="hybrid")
    verifier  optional Verifier: screens the top-k before the agent sees them, and its reasons go with each
              note. Its token usage is added to RunState.verifier_* (kept separate from the agent's).
    """
    if backend not in ("bm25", "hybrid"):
        raise ValueError(f"unknown backend {backend!r}")

    def handler(args: Dict[str, Any], state: RunState) -> ToolResult:
        core, aux = _check_keywords(args)
        query = ""
        if backend == "hybrid":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty natural-language string")
            query = query.strip()
            ranked = searcher.search(core, aux, query)                          # fused, best first
            top = [pid for pid, _ in ranked[:top_k]]
            candidates = top
        else:
            ranked = searcher.search(core, aux, aux_weight=aux_weight)          # all patients
            top = [pid for pid, _ in ranked[:top_k]]
            candidates = [pid for pid, s in ranked[:top_k] if s > 0]           # zero-score notes never match

        call = SearchCall(core, aux, ranked, top, query=query, backend=backend, shown_ids=candidates)
        if verifier is not None and candidates:
            verdicts, usage = verifier.verify(state.query, {pid: searcher.get_note(pid) for pid in candidates})
            state.add_verifier_usage(usage)
            call.verdicts = [dataclasses.asdict(v) for v in verdicts]
            call.shown_ids = [v.patient_id for v in verdicts if v.kept]
            state.log("verify", n_in=len(candidates), n_kept=len(call.shown_ids), errors=usage.errors,
                      labels={l: sum(v.label == l for v in verdicts)
                              for l in ("relevant", "uncertain", "not_relevant", "error")},
                      seconds=usage.seconds)
        elif verifier is not None:
            call.verdicts = []
        return ToolResult(output=format_observation(searcher, call), artifact=call)

    schema = make_hybrid_schema(top_k) if backend == "hybrid" else make_bm25_schema(top_k)
    return Tool(name=schema["name"], schema=schema, handler=handler, max_calls=max_calls)


def make_bm25_tool(bm25: BM25Search, top_k: int = 20, aux_weight: float = 0.5,
                   max_calls: Optional[int] = 1, verifier: Any = None) -> Tool:
    """Backward-compatible shortcut for the keyword-only tool."""
    return make_search_tool(bm25, "bm25", top_k, aux_weight, max_calls, verifier)
