import pytest

from src import agent
from src.agent import MAX_STEPS, Trace, calculator, knowledge_base, parse, solve


def test_calculator_evaluates():
    assert calculator("128 * 47") == "6016"


def test_calculator_refuses_code():
    assert calculator("__import__('os').system('ls')").startswith("error")


def test_knowledge_base_miss():
    assert knowledge_base("quantum tunnelling") == "no entry found"


def test_parse_action():
    step = parse("Thought: need math\nAction: calculator\nAction Input: 2+2")
    assert step.action == "calculator" and step.action_input == "2+2"


def test_parse_final_answer():
    step = parse("Thought: done\nFinal Answer: 42")
    assert step.action is None and step.thought == "42"


def test_solve_reaches_answer(monkeypatch):
    replies = iter([
        "Thought: math\nAction: calculator\nAction Input: 2*3",
        "Thought: done\nFinal Answer: it is 6",
    ])
    monkeypatch.setattr(agent, "complete", lambda m, **k: next(replies))
    t = solve("2*3?")
    assert t.answer == "it is 6" and t.stopped_because == "finished"


def test_loop_detection_breaks_out(monkeypatch):
    monkeypatch.setattr(
        agent, "complete",
        lambda m, **k: "Thought: again\nAction: calculator\nAction Input: 1+1",
    )
    t = solve("loop please")
    assert t.stopped_because == "repeated_action"
    assert len(t.steps) < MAX_STEPS


def test_step_cap_enforced(monkeypatch):
    n = {"i": 0}

    def alternating(messages, **kwargs):
        n["i"] += 1
        return f"Thought: t\nAction: calculator\nAction Input: {n['i']}+1"

    monkeypatch.setattr(agent, "complete", alternating)
    t = solve("never finish", max_steps=4)
    assert len(t.steps) == 4 and t.stopped_because == "step_limit"


def test_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setattr(
        agent, "complete",
        lambda m, **k: f"Thought: t\nAction: calculator\nAction Input: {id(m) % 7}+1",
    )
    t = solve("never finish", max_steps=2)
    assert t.answer and "Partial answer" in t.answer


def test_timeout_stops(monkeypatch):
    monkeypatch.setattr(
        agent, "complete",
        lambda m, **k: "Thought: t\nAction: calculator\nAction Input: 1+2",
    )
    t = solve("slow", max_seconds=-1)
    assert t.stopped_because == "timeout"


def test_trace_renders():
    from src.agent import Step

    t = Trace(steps=[Step(thought="think", action="calculator", observation="4")])
    assert "calculator" in t.render()
