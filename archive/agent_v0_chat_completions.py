"""
Single-agent search + reasoning over Asclepius notes  (v0: one tool, BM25)
===========================================================================

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
REASONING_EFFORT = "none"  # Chat Completions + function tools on gpt-6-luna requires "none".
                           # Reasoning + tools needs the Responses API (/v1/responses).
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


BM25_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
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
    messages: List[Dict[str, Any]] = field(default_factory=list)   # full trace, for debugging
    n_llm_calls: int = 0

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

    def _llm(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None, **kwargs):
        """One chat-completions call. Returns the assistant message object.
        Extra kwargs (e.g. tool_choice="none") are passed straight to the API."""
        kwargs.update({"model": self.model, "messages": messages})
        if REASONING_EFFORT is not None:
            kwargs.setdefault("reasoning_effort", REASONING_EFFORT)
        if tools:
            kwargs["tools"] = tools
        return client.chat.completions.create(**kwargs).choices[0].message

    def _log(self, *a):
        if self.verbose:
            print(*a)

    def run(self, query: str, max_searches: int = 1) -> AgentResult:
        """
        The agent loop:

          messages = [system, user(query)]
          repeat up to max_steps:
              msg = LLM(messages, tools)                 # tools are disabled once the search budget is used
              append the assistant turn to messages      # BEFORE any tool results
              if no tool_calls: msg.content is the final answer -> return
              else: for EACH tool call: run it, append {"role": "tool", "tool_call_id": ..., "content": ...}
          out of steps -> one last call with tools disabled, to force an answer

        max_searches=1 matches the design "search once, then answer". Raise it to let the agent re-search.
        """
        result = AgentResult(query=query)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)},
        ]
        result.messages = messages            # same list object, so the trace stays up to date
        self._log(f"\nQUERY: {query}")

        for step in range(1, self.max_steps + 1):
            can_search = len(result.searches) < max_searches
            # tool_choice="none": the model still sees the tool's definition (needed because the history
            # contains tool calls), but it is not allowed to call it, so it has to answer.
            msg = self._llm(messages, tools=[BM25_TOOL_SCHEMA], tool_choice="auto" if can_search else "none")
            result.n_llm_calls += 1
            messages.append(self._assistant_turn(msg))

            if not msg.tool_calls:                         # no tool call -> this is the final answer
                result.answer = msg.content
                self._log(f"[step {step}] final answer ({len(msg.content or '')} chars)")
                return result

            for call in msg.tool_calls:                    # every tool_call.id needs a matching tool message
                content = self._execute_tool_call(call, result, max_searches)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

        # Ran out of steps without an answer: force one, with tools disabled.
        self._log(f"[max_steps={self.max_steps} reached] forcing a final answer")
        msg = self._llm(messages, tools=[BM25_TOOL_SCHEMA], tool_choice="none")
        result.n_llm_calls += 1
        messages.append(self._assistant_turn(msg))
        result.answer = msg.content
        return result

    @staticmethod
    def _assistant_turn(msg) -> Dict[str, Any]:
        """The SDK message object -> a plain dict we can send back to the API.
        Built by hand (instead of msg.model_dump()) so only fields the API accepts as input are included."""
        turn: Dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            turn["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in msg.tool_calls
            ]
        return turn

    def _execute_tool_call(self, call, result: AgentResult, max_searches: int) -> str:
        """Run one tool call and return the text for its tool message.
        Errors are returned as text, not raised: the model reads them and can correct itself."""
        name = call.function.name
        if name != "bm25_search":
            return f"Error: unknown tool '{name}'. The only tool is bm25_search."
        if len(result.searches) >= max_searches:           # e.g. two parallel calls when the budget is 1
            return "Search budget used up. Answer using the notes already retrieved."
        try:
            args = json.loads(call.function.arguments)
            core = [str(k) for k in args["core_keywords"]]
            aux = [str(k) for k in args.get("auxiliary_keywords", [])]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self._log(f"  bad tool arguments: {call.function.arguments!r} ({e})")
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
    print(f"LLM calls: {res.n_llm_calls}")
    print("\nANSWER:\n", res.answer)
    print("\nGROUNDING:", check_grounding(res))
