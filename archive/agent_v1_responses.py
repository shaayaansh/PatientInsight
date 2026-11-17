"""
Single-agent search + reasoning over Asclepius notes  (v1: one tool, BM25, Responses API)
=========================================================================================
(v0, the Chat Completions version, is kept in archive/agent_v0_chat_completions.py)

The agent is both planner and executor. It has ONE tool: `bm25_search`.

    query
      │
      ▼
  ┌────────┐  tool call: bm25_search(core_keywords, auxiliary_keywords)
  │  LLM   │ ───────────────────────────────────────────────────────────┐
  │ (agent)│                                                            ▼
  │        │ ◀── observation: top-K (=20) notes, with patient ids ── BM25 over ALL patients
  └────────┘
      │  final answer, grounded ONLY in the K notes, citing patient ids
      ▼

Assignment-style: the code marked `YOUR CODE HERE` is yours to write. Everything else
is plumbing you shouldn't need to change (but read it).

Parts
  Part 1  prompts/system.md             the agent's role and rules (written for you: tweak and experiment)
  Part 2  BM25_TOOL_SCHEMA              how the LLM "sees" the tool (JSON schema)
  Part 3  format_observation()          how the tool result is shown to the LLM
  Part 4  Agent.run()                   the agent loop: LLM -> tool -> LLM -> answer
  Part 5  check_grounding()             verify the answer only cites retrieved patients

Run:   python agent.py            (runs the agent on the first query in the queries file)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from bm25_search import BM25Search

# --------------------------------------------------------------------------- config
MODEL = "gpt-6-luna"
TOP_K = 20               # how many patients the agent is allowed to read
REASONING_EFFORT = "low"    # "none" | "low" | "medium" | "high"  (reasoning tokens are billed as output)
REASONING_SUMMARY = "auto"  # ask for a readable summary of the reasoning (for logging). None to disable;
                            # some accounts need organization verification for summaries.
AUX_WEIGHT = 0.5         # weight of auxiliary keywords in the BM25 score
MAX_STEPS = 4            # safety cap on LLM calls in one run
QUERIES_PATH = Path("data/asclepius/general_temporal_queries/temporal_information_queries.jsonl")

client = OpenAI(api_key=Path("openai_secret.txt").read_text().strip())


# =========================================================================== Part 1  (prompt text lives in prompts/system.md)
# The system prompt defines the agent's role and rules. It should at least cover:
#   - what the agent is (clinical search + reasoning assistant over a corpus of discharge summaries)
#   - the tool is KEYWORD search (BM25): no semantic matching, so it must turn the question into
#     core keywords (from the query) AND auxiliary keywords (synonyms, abbreviations, related tests /
#     treatments / complications that would appear in notes that answer the question)
#   - it will only see the top-K notes, and must answer ONLY from them, never from its own knowledge
#   - temporal focus: order events, say what came before/after, and point out when notes disagree
#   - citations: every claim cites the patient(s) it came from, in a fixed format, e.g. [patient 12345]
#   - what to do if the notes don't contain the answer (say so, don't make it up)
#
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_FILE = "system.md"     # swap in a variant here, e.g. "system_v2.md"


def load_prompt(name: str, **kwargs) -> str:
    """Read prompts/<name> and fill {placeholders} (e.g. {top_k}) with str.format.
    Note: a literal { or } in the .md file must be doubled ({{ }})."""
    return (PROMPTS_DIR / name).read_text().strip().format(**kwargs)


SYSTEM_PROMPT = load_prompt(SYSTEM_PROMPT_FILE, top_k=TOP_K)

USER_PROMPT_TEMPLATE = "Question: {query}" 


# Responses API format: name/description/parameters sit at the top level
# (Chat Completions wrapped them in "function": {...}).
BM25_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "name": "bm25_search",
    "description": (
        f"Keyword (BM25) search over all patients' discharge summaries. Matches exact words after "
        f"stemming; no semantic matching. Returns the top {TOP_K} notes, each labeled with its patient id."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "core_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key clinical entities taken directly from the question "
                               "(conditions, procedures, drugs, tests). Short and specific.",
            },
            "auxiliary_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Expansion terms not in the question that are likely to appear in "
                               "relevant notes: synonyms, abbreviations, related tests, treatments, "
                               "complications. Weighted lower than core keywords.",
            },
        },
        "required": ["core_keywords", "auxiliary_keywords"],
        "additionalProperties": False,
    },
}


# =========================================================================== tool execution
@dataclass
class SearchCall:
    core_keywords: List[str]
    auxiliary_keywords: List[str]
    ranked: List[Tuple[int, float]]          # ALL patients, (patient_id, score), sorted
    top_k_ids: List[int]                     # the K patients the agent actually got to read


def run_bm25_tool(bm25: BM25Search, core_keywords: List[str], auxiliary_keywords: List[str],
                  top_k: int = TOP_K) -> SearchCall:
    ranked = bm25.search(core_keywords, auxiliary_keywords, aux_weight=AUX_WEIGHT)   # all patients
    return SearchCall(core_keywords, auxiliary_keywords, ranked, [pid for pid, _ in ranked[:top_k]])


# =========================================================================== Part 3
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

# =========================================================================== results (given)
@dataclass
class AgentResult:
    query: str
    answer: Optional[str] = None
    searches: List[SearchCall] = field(default_factory=list)
    items: List[Any] = field(default_factory=list)   # full conversation (input + output items), for debugging
    n_llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0                           # includes reasoning tokens
    reasoning_tokens: int = 0

    @property
    def retrieved_ids(self) -> List[int]:
        """Every patient the agent was shown (across all searches)."""
        seen: List[int] = []
        for s in self.searches:
            seen += [p for p in s.top_k_ids if p not in seen]
        return seen


# =========================================================================== Part 4
class Agent:
    def __init__(self, bm25: BM25Search, model: str = MODEL, top_k: int = TOP_K,
                 max_steps: int = MAX_STEPS, verbose: bool = True):
        self.bm25, self.model, self.top_k = bm25, model, top_k
        self.max_steps, self.verbose = max_steps, verbose

    def _llm(self, items: List[Any], tools: Optional[List[Dict]] = None, **kwargs):
        """One Responses API call. Returns the whole Response object
        (we need response.output for the items and response.usage for cost).
        Extra kwargs (e.g. tool_choice="none") are passed straight to the API."""
        kwargs.update({"model": self.model, "instructions": SYSTEM_PROMPT, "input": items})
        if REASONING_EFFORT is not None:
            reasoning = {"effort": REASONING_EFFORT}
            if REASONING_SUMMARY:
                reasoning["summary"] = REASONING_SUMMARY
            kwargs.setdefault("reasoning", reasoning)
        if tools:
            kwargs["tools"] = tools
        return client.responses.create(**kwargs)

    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _record(self, response, result: AgentResult, step: int):
        """Book-keeping after every LLM call: counters, token usage, reasoning summary."""
        result.n_llm_calls += 1
        u = getattr(response, "usage", None)
        if u is not None:
            result.input_tokens += u.input_tokens or 0
            result.output_tokens += u.output_tokens or 0
            details = getattr(u, "output_tokens_details", None)
            result.reasoning_tokens += (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        if getattr(response, "status", "completed") == "incomplete":
            self._log(f"[step {step}] WARNING: response incomplete: {getattr(response, 'incomplete_details', None)}")
        for item in response.output:
            if item.type == "reasoning":
                summary = " ".join(getattr(s_, "text", "") for s_ in (getattr(item, "summary", None) or []))
                if summary:
                    self._log(f"[step {step}] reasoning: {summary[:600]}")

    def run(self, query: str, max_searches: int = 1) -> AgentResult:
        """
        The agent loop (Responses API):

          items = [user(query)]                      # the system prompt goes in `instructions=`, not here
          repeat up to max_steps:
              response = LLM(items, tools)           # tools are disabled once the search budget is used
              items += response.output               # ALL output items, reasoning included, BEFORE tool results
              function_calls = output items with type == "function_call"
              if none: response.output_text is the final answer -> return
              else: for EACH call: run it, append {"type": "function_call_output", "call_id": ..., "output": ...}
          out of steps -> one last call with tools disabled, to force an answer

        max_searches=1 matches the design "search once, then answer". Raise it to let the agent re-search.
        """
        result = AgentResult(query=query)
        items: List[Any] = [{"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)}]
        result.items = items                  # same list object, so the trace stays up to date
        self._log(f"\nQUERY: {query}")

        for step in range(1, self.max_steps + 1):
            can_search = len(result.searches) < max_searches
            # tool_choice="none": the model still sees the tool's definition, but may not call it.
            response = self._llm(items, tools=[BM25_TOOL_SCHEMA], tool_choice="auto" if can_search else "none")
            self._record(response, result, step)

            # Send the output back unchanged next turn. Reasoning items must be included, otherwise
            # the model loses its earlier chain of thought between the search and the answer.
            items.extend(response.output)

            calls = [it for it in response.output if it.type == "function_call"]
            if not calls:                                  # no tool call -> this is the final answer
                result.answer = response.output_text
                self._log(f"[step {step}] final answer ({len(result.answer or '')} chars)")
                return result

            for call in calls:                             # every call_id needs a matching output item
                output = self._execute_tool_call(call, result, max_searches)
                items.append({"type": "function_call_output", "call_id": call.call_id, "output": output})

        # Ran out of steps without an answer: force one, with tools disabled.
        self._log(f"[max_steps={self.max_steps} reached] forcing a final answer")
        response = self._llm(items, tools=[BM25_TOOL_SCHEMA], tool_choice="none")
        self._record(response, result, self.max_steps + 1)
        items.extend(response.output)
        result.answer = response.output_text
        return result

    def _execute_tool_call(self, call, result: AgentResult, max_searches: int) -> str:
        """Run one function_call item and return the text for its function_call_output.
        Note: use call.call_id (links call <-> output), NOT call.id (the item's own id).
        Errors are returned as text, not raised: the model reads them and can correct itself."""
        if call.name != "bm25_search":
            return f"Error: unknown tool '{call.name}'. The only tool is bm25_search."
        if len(result.searches) >= max_searches:           # e.g. two parallel calls when the budget is 1
            return "Search budget used up. Answer using the notes already retrieved."
        try:
            args = json.loads(call.arguments)
            core = [str(k) for k in args["core_keywords"]]
            aux = [str(k) for k in args.get("auxiliary_keywords", [])]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self._log(f"  bad tool arguments: {call.arguments!r} ({e})")
            return (f"Error: could not parse arguments ({e}). Call bm25_search with JSON "
                    f'{{"core_keywords": [...], "auxiliary_keywords": [...]}}.')

        search = run_bm25_tool(self.bm25, core, aux, self.top_k)
        result.searches.append(search)
        self._log(f"  bm25_search  core={core}\n               aux={aux}")
        self._log(f"  -> showing top {len(search.top_k_ids)}: {search.top_k_ids}")
        return format_observation(self.bm25, search)


# =========================================================================== Part 5  (implemented)
# The prompt asks for "[patient 12345]", but models drift: "[patient 123, patient 456]",
# "[patients 123, 456]", "[Patient 123; 456]". So: find every [...] group that mentions
# "patient", then take every number inside it.
CITATION_GROUP_RE = re.compile(r"\[([^\[\]]*?\bpatients?\b[^\[\]]*?)\]", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+")


def _shown_ids(result: AgentResult) -> List[int]:
    """Patients whose note text was actually in the context. format_observation drops
    zero-score notes, so this can be fewer than top_k_ids."""
    shown: List[int] = []
    for s in result.searches:
        scores = dict(s.ranked[:len(s.top_k_ids)])
        shown += [p for p in s.top_k_ids if scores.get(p, 0) > 0 and p not in shown]
    return shown


def check_grounding(result: AgentResult) -> Dict[str, Any]:
    """
    A cheap, automatic grounding check on the final answer:
      - which patient ids does the answer cite?
      - are any of them NOT among the notes the model was shown? (= hallucinated citations)
      - does the answer cite anything at all?

    This only checks WHO is cited, not whether the claim is actually in that note.
    That needs an LLM judge, which is a later step.
    """
    answer = result.answer or ""
    shown = _shown_ids(result)
    shown_set = set(shown)

    mentions: List[int] = []                          # every citation, in order, with repeats
    for group in CITATION_GROUP_RE.findall(answer):
        mentions += [int(n) for n in NUMBER_RE.findall(group)]
    cited = list(dict.fromkeys(mentions))             # unique, in first-cited order

    hallucinated = [p for p in cited if p not in shown_set]
    return {
        "has_answer": bool(answer.strip()),
        "has_citations": bool(cited),
        "n_citation_mentions": len(mentions),
        "cited": cited,
        "hallucinated": hallucinated,                     # cited but never shown -> made up
        "n_shown": len(shown),
        "frac_shown_cited": round(len(set(cited) & shown_set) / len(shown), 3) if shown else 0.0,
        "grounded": bool(cited) and not hallucinated,     # cites something, and only what it was shown
    }


# =========================================================================== main (given)
if __name__ == "__main__":
    queries = [json.loads(l) for l in open(QUERIES_PATH)]
    item = queries[0]

    bm25 = BM25Search.load_or_build()
    agent = Agent(bm25)
    res = agent.run(item["query"])

    print("\n" + "=" * 100)
    print("QUERY:  ", item["query"])
    for i, s in enumerate(res.searches, 1):
        print(f"SEARCH {i}: core={s.core_keywords}\n          aux={s.auxiliary_keywords}")
        gold_rank = next((r for r, (p, _) in enumerate(s.ranked, 1) if p == item["patient_id"]), None)
        print(f"          gold patient {item['patient_id']} rank = {gold_rank}  (in top-{TOP_K}: {item['patient_id'] in s.top_k_ids})")
    print(f"LLM calls: {res.n_llm_calls} | tokens in={res.input_tokens:,} out={res.output_tokens:,} "
          f"(reasoning={res.reasoning_tokens:,})")
    print("\nANSWER:\n", res.answer)
    print("\nGROUNDING:", check_grounding(res))
