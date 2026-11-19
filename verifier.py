"""
Verifier: screens retrieved notes for relevance BEFORE the main agent reads them.

    search tool:  retrieve top-k  ──▶  Verifier.verify(question, notes)  ──▶  only kept notes (+ a reason
                                        one LLM call per note, in parallel     each) go into the observation

It is not an agent: no tools, no loop, no decisions about what to do next. It's a classifier with a
swappable model. Everything about it lives in VerifierConfig, so moving to a smaller or local model is a
config change:

    VerifierConfig(model="gpt-5.4-nano")                                        # OpenAI, Responses API
    VerifierConfig(model="Qwen/Qwen3-8B", base_url="http://localhost:8000/v1",  # local vLLM server
                   api="chat", reasoning_effort=None)                           # (OpenAI-compatible)

Each verdict has a label (relevant / uncertain / not_relevant / error) and a one-sentence reason that the
main agent gets with the note: it says WHAT in the note bears on the question, so the agent finds it fast.
(Later the reason becomes structured fields: that's the schema-extraction step.)

Cost is tracked separately from the main agent: verify() returns its token usage, and the tool adds it to
RunState.verifier_* so a verified run is never reported as cheaper than it really was.
"""
from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
LABELS = ["relevant", "uncertain", "not_relevant"]
RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
             openai.InternalServerError)


@dataclass
class VerifierConfig:
    model: str = "gpt-5.4-nano"
    api: str = "responses"                    # "responses" (OpenAI) | "chat" (Chat Completions: vLLM, most local servers)
    base_url: Optional[str] = None            # None = OpenAI. e.g. "http://localhost:8000/v1" for vLLM
    api_key: Optional[str] = None             # None = read openai_secret.txt (or "EMPTY" when base_url is set)
    prompt_file: str = "verify_relevance.md"  # in prompts/
    reasoning_effort: Optional[str] = "low"   # None for models without a reasoning setting (most local models)
    max_concurrency: int = 10                 # notes verified in parallel
    keep_uncertain: bool = True               # pass borderline notes to the agent (a wrongly dropped note is gone for good)
    on_error: str = "keep"                    # "keep" | "drop": what to do with a note whose verification failed
    max_note_words: int = 3000                # very long notes are cut before verification (the agent still gets the full note)
    max_tries: int = 4                        # per note, for transient API errors


@dataclass
class Verdict:
    patient_id: int
    label: str                                # relevant | uncertain | not_relevant | error
    reason: str
    kept: bool                                # does the agent get to read this note?


@dataclass
class VerifierUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    errors: int = 0
    seconds: float = 0.0


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},         # first, so the label is conditioned on it
        "label": {"type": "string", "enum": LABELS},
    },
    "required": ["reason", "label"],
    "additionalProperties": False,
}


class Verifier:
    def __init__(self, config: Optional[VerifierConfig] = None, client: Any = None):
        self.config = config or VerifierConfig()
        self.instructions = (ROOT / "prompts" / self.config.prompt_file).read_text().strip()
        self.client = client or self._make_client()
        self._lock = threading.Lock()

    def _make_client(self):
        c = self.config
        if c.base_url:
            return OpenAI(base_url=c.base_url, api_key=c.api_key or "EMPTY", max_retries=0)
        key = c.api_key or (ROOT / "openai_secret.txt").read_text().strip()
        return OpenAI(api_key=key, max_retries=0)

    # ------------------------------------------------------------------ public
    def verify(self, question: str, notes: Dict[int, str]) -> tuple[List[Verdict], VerifierUsage]:
        """Verify every note (in parallel). Returns verdicts in the SAME order as `notes`, plus usage."""
        usage = VerifierUsage()
        if not notes:
            return [], usage
        t0 = time.time()
        items = list(notes.items())
        with ThreadPoolExecutor(max_workers=min(self.config.max_concurrency, len(items))) as pool:
            verdicts = list(pool.map(lambda kv: self._verify_one(question, kv[0], kv[1], usage), items))
        usage.seconds = round(time.time() - t0, 2)
        return verdicts, usage

    def describe(self) -> Dict[str, Any]:
        """Config for config.json (no API key)."""
        d = asdict(self.config)
        d.pop("api_key", None)
        return d

    # ------------------------------------------------------------------ internals
    def _verify_one(self, question: str, pid: int, note: str, usage: VerifierUsage) -> Verdict:
        words = note.split()
        if len(words) > self.config.max_note_words:
            note = " ".join(words[:self.config.max_note_words]) + " [... truncated ...]"
        user = f"Question: {question}\n\n<note patient_id=\"{pid}\">\n{note.strip()}\n</note>"
        try:
            out = self._call(user, usage)
            label = out["label"] if out.get("label") in LABELS else "uncertain"
            reason = " ".join(str(out.get("reason", "")).split())
        except Exception as e:                      # never let one note break the search
            with self._lock:
                usage.errors += 1
            return Verdict(pid, "error", f"verification failed ({type(e).__name__})",
                           kept=self.config.on_error == "keep")
        kept = label == "relevant" or (label == "uncertain" and self.config.keep_uncertain)
        return Verdict(pid, label, reason, kept)

    def _call(self, user: str, usage: VerifierUsage) -> Dict[str, Any]:
        c = self.config
        for attempt in range(c.max_tries):
            try:
                if c.api == "responses":
                    kwargs: Dict[str, Any] = dict(
                        model=c.model, instructions=self.instructions, input=user,
                        text={"format": {"type": "json_schema", "name": "verdict",
                                         "schema": VERDICT_SCHEMA, "strict": True}})
                    if c.reasoning_effort:
                        kwargs["reasoning"] = {"effort": c.reasoning_effort}
                    resp = self.client.responses.create(**kwargs)
                    self._add_usage(usage, resp, "responses")
                    return json.loads(resp.output_text)
                else:   # chat completions: what vLLM and most local servers speak
                    kwargs = dict(
                        model=c.model,
                        messages=[{"role": "system", "content": self.instructions},
                                  {"role": "user", "content": user}],
                        response_format={"type": "json_schema", "json_schema": {
                            "name": "verdict", "schema": VERDICT_SCHEMA, "strict": True}})
                    if c.reasoning_effort:
                        kwargs["reasoning_effort"] = c.reasoning_effort
                    resp = self.client.chat.completions.create(**kwargs)
                    self._add_usage(usage, resp, "chat")
                    return json.loads(resp.choices[0].message.content)
            except RETRYABLE:
                if attempt == c.max_tries - 1:
                    raise
                time.sleep(min(30, 2 ** attempt) + random.random())
        raise RuntimeError("unreachable")

    def _add_usage(self, usage: VerifierUsage, resp: Any, api: str) -> None:
        u = getattr(resp, "usage", None)
        with self._lock:
            usage.calls += 1
            if u is None:
                return
            if api == "responses":
                usage.input_tokens += getattr(u, "input_tokens", 0) or 0
                usage.output_tokens += getattr(u, "output_tokens", 0) or 0
                d = getattr(u, "output_tokens_details", None)
            else:
                usage.input_tokens += getattr(u, "prompt_tokens", 0) or 0
                usage.output_tokens += getattr(u, "completion_tokens", 0) or 0
                d = getattr(u, "completion_tokens_details", None)
            usage.reasoning_tokens += (getattr(d, "reasoning_tokens", 0) or 0) if d else 0
