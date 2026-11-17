"""
Autograder for orchestrator.py: runs your Orchestrator against a FAKE model (scripted replies),
so it's free and deterministic. No API calls are made.

    python test_orchestrator.py
"""
import json
import tempfile
import traceback
import types
from pathlib import Path

import agent
import orchestrator as O
from state import RunStatus

NS = types.SimpleNamespace
GOOD = json.dumps({"core_keywords": ["congestive heart failure"], "auxiliary_keywords": ["BNP", "echocardiogram"]})
Q = "What symptoms and tests lead to a diagnosis of congestive heart failure?"


def fc(cid, args, name="bm25_search"):
    return NS(type="function_call", id="fc_" + cid, call_id=cid, name=name, arguments=args, status="completed")

def rs(text):
    return NS(type="reasoning", id="rs_1", summary=[NS(type="summary_text", text=text)])

def msg(text):
    return NS(type="message", role="assistant", content=[NS(type="output_text", text=text)])

def resp(*out, text=""):
    return NS(output=list(out), output_text=text, status="completed",
              usage=NS(input_tokens=100, output_tokens=50, output_tokens_details=NS(reasoning_tokens=20)))


class FakeModel:
    """Replaces the OpenAI client. Returns scripted responses and records every request."""
    def __init__(self, script, fail_on_call=None):
        self.script, self.fail_on_call, self.requests = list(script), fail_on_call, []
        self.responses = NS(create=self.create)

    def create(self, **kwargs):
        kwargs = dict(kwargs, input=list(kwargs["input"]))   # snapshot: the orchestrator keeps appending to the same list
        self.requests.append(kwargs)
        if self.fail_on_call == len(self.requests):
            raise RuntimeError("API down")
        if not self.script:
            raise AssertionError("the orchestrator made more LLM calls than the test expected")
        return self.script.pop(0)

    def input_types(self, i):
        return [it.get("type") or it.get("role") for it in self.requests[i]["input"]]


def _orch():
    return O.Orchestrator(agent.Agent(), _orch.registry, max_steps=O.MAX_STEPS, checkpoint_dir=None, verbose=False)


def outputs(state):
    return [it for it in state.items if it.get("type") == "function_call_output"]


# ------------------------------------------------------------------ tests
def test_normal_run():
    fake = agent.client = FakeModel([resp(rs("pick terms"), fc("c1", GOOD)), resp(msg("A [patient 1]"), text="A [patient 1]")])
    s = _orch().run(Q)
    assert s.status is RunStatus.ANSWERED, f"{s.status} (state.error = {s.error})"
    assert s.answer == "A [patient 1]"
    assert len(s.searches) == 1 and s.tool_calls == {"bm25_search": 1}
    assert fake.requests[0]["tool_choice"] == "auto"
    assert fake.requests[1]["tool_choice"] == "none", "the search limit (1) is used up, so tools must be off on call 2"
    assert fake.input_types(1) == ["user", "reasoning", "function_call", "function_call_output"], fake.input_types(1)
    assert outputs(s)[0]["call_id"] == "c1", "function_call_output must use call_id, not the item id"
    assert s.n_llm_calls == 2 and s.input_tokens == 200 and s.reasoning_tokens == 40
    assert s.finished_at is not None

def test_bad_json_is_reported_and_not_counted():
    agent.client = FakeModel([resp(fc("c1", "{bad")), resp(fc("c2", GOOD)), resp(msg("ans"), text="ans")])
    s = _orch().run(Q)
    assert s.status is RunStatus.ANSWERED
    assert s.tool_calls == {"bm25_search": 1}, "bad JSON should not use up the search limit"
    assert outputs(s)[0]["output"].lower().startswith("error"), "the model should get an error message back"

def test_unknown_tool():
    agent.client = FakeModel([resp(fc("c1", "{}", name="read_note")), resp(msg("ans"), text="ans")])
    s = _orch().run(Q)
    assert s.status is RunStatus.ANSWERED
    assert "error" in outputs(s)[0]["output"].lower() and s.tool_calls.get("read_note", 0) == 0

def test_handler_rejects_args():
    agent.client = FakeModel([resp(fc("c1", json.dumps({"core_keywords": [], "auxiliary_keywords": []}))),
                              resp(msg("ans"), text="ans")])
    s = _orch().run(Q)
    assert "error" in outputs(s)[0]["output"].lower() and len(s.searches) == 0

def test_parallel_calls_each_get_an_output():
    agent.client = FakeModel([resp(fc("c1", GOOD), fc("c2", GOOD)), resp(msg("ans"), text="ans")])
    s = _orch().run(Q)
    assert [o["call_id"] for o in outputs(s)] == ["c1", "c2"], "every function_call needs its own output"
    assert len(s.searches) == 1, "only one search is allowed; the second call must be refused"

def test_step_limit_forces_answer():
    bad = lambda i: resp(fc(f"c{i}", "{bad"))
    fake = agent.client = FakeModel([bad(i) for i in range(O.MAX_STEPS)] + [resp(msg("forced"), text="forced")])
    s = _orch().run(Q)
    assert s.status is RunStatus.ANSWERED and s.answer == "forced"
    assert fake.requests[-1]["tool_choice"] == "none", "the forced final call must have tools disabled"
    assert len(fake.requests) == O.MAX_STEPS + 1
    assert any(e.kind == "forced_answer" for e in s.events), "log a 'forced_answer' event"

def test_api_error_does_not_crash():
    agent.client = FakeModel([resp(fc("c1", GOOD))], fail_on_call=2)
    s = _orch().run(Q)
    assert s.status is RunStatus.ERROR and "API down" in (s.error or "")
    assert s.finished_at is not None

def test_checkpoint_written():
    agent.client = FakeModel([resp(fc("c1", GOOD)), resp(msg("ans [patient 1]"), text="ans [patient 1]")])
    with tempfile.TemporaryDirectory() as d:
        o = _orch(); o.checkpoint_dir = Path(d)
        s = o.run(Q)
        saved = json.loads((Path(d) / f"{s.run_id}.json").read_text())
        assert saved["status"] == "answered" and saved["answer"] == s.answer


if __name__ == "__main__":
    print("loading BM25 index ...")
    _orch.registry = O.build().registry
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); passed += 1; print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception:
            print(f"ERROR {t.__name__}:\n{traceback.format_exc()}")
    print(f"\n{passed}/{len(tests)} passed")
