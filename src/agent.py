"""ReAct Planning Agent - observe, think, act, with a hard stop.

The loop is explicit state rather than `while True`, bounded by both a step
count and a wall clock, and it detects the classic failure where a model asks
for the same tool with the same arguments forever.
"""
from __future__ import annotations

import ast
import operator
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .llm import complete
from .logging_setup import log

MAX_STEPS = 6
MAX_SECONDS = 30.0
DEMO = "What is 128 * 47, and what does the knowledge base say about retries?"

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def calculator(expr: str) -> str:
    """Arithmetic without eval()."""

    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("unsupported expression")

    try:
        return str(walk(ast.parse(expr.strip(), mode="eval").body))
    except Exception as err:  # noqa: BLE001
        return f"error: {err}"


KB = {
    "retries": "The agent retries up to three times before giving up.",
    "budget": "Token budgets are enforced before the call, not after.",
    "memory": "Memory has a short-term buffer and a long-term vector store.",
}


def knowledge_base(query: str) -> str:
    hits = [v for k, v in KB.items() if k in query.lower()]
    return " ".join(hits) if hits else "no entry found"


TOOLS: dict[str, Callable[[str], str]] = {
    "calculator": calculator,
    "knowledge_base": knowledge_base,
}


@dataclass
class Step:
    thought: str = ""
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


@dataclass
class Trace:
    steps: list[Step] = field(default_factory=list)
    answer: str | None = None
    stopped_because: str = "finished"

    def render(self) -> str:
        lines = []
        for i, s in enumerate(self.steps, 1):
            lines.append(f"{i}. thought: {s.thought}")
            if s.action:
                lines.append(f"   action: {s.action}({s.action_input})  ->  {s.observation}")
        return "\n".join(lines)


SYSTEM = """You solve tasks with tools, one step at a time.

Tools:
- calculator(expression)
- knowledge_base(query)

Reply in exactly this format:

Thought: <your reasoning>
Action: <tool name>
Action Input: <argument>

When you can answer, reply instead with:

Thought: <your reasoning>
Final Answer: <the answer>"""


def parse(raw: str) -> Step:
    """Pull Thought / Action / Action Input / Final Answer out of the reply."""
    def grab(label: str) -> str | None:
        m = re.search(rf"{label}\s*:\s*(.+?)(?=\n[A-Z][A-Za-z ]*:|\Z)", raw, re.S)
        return m.group(1).strip() if m else None

    final = grab("Final Answer")
    thought = grab("Thought") or raw.strip()
    if final is not None:
        return Step(thought=final)
    return Step(thought=thought, action=grab("Action"), action_input=grab("Action Input"))


def solve(question: str, max_steps: int = MAX_STEPS, max_seconds: float = MAX_SECONDS) -> Trace:
    trace = Trace()
    started = time.monotonic()
    scratch = ""

    for i in range(max_steps):
        if time.monotonic() - started > max_seconds:
            trace.stopped_because = "timeout"
            break

        step = parse(complete([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{question}\n{scratch}"},
        ]))

        if step.action is None:
            step.observation = None
            trace.steps.append(step)
            trace.answer = step.thought
            return trace

        prev = trace.steps[-1] if trace.steps else None
        if prev and (prev.action, prev.action_input) == (step.action, step.action_input):
            log.warning("repeated_action", extra={"action": step.action, "step": i})
            trace.stopped_because = "repeated_action"
            break

        tool = TOOLS.get(step.action)
        step.observation = tool(step.action_input or "") if tool else f"unknown tool: {step.action}"
        trace.steps.append(step)
        scratch += (
            f"\nThought: {step.thought}\nAction: {step.action}"
            f"\nAction Input: {step.action_input}\nObservation: {step.observation}"
        )
    else:
        trace.stopped_because = "step_limit"

    if trace.answer is None:
        trace.answer = degrade(question, trace)
    return trace


def degrade(question: str, trace: Trace) -> str:
    """Best effort from whatever was observed, instead of an exception."""
    observed = [s.observation for s in trace.steps if s.observation]
    if not observed:
        return "I could not make progress on that."
    return "Partial answer from what I gathered: " + " | ".join(observed)


def run(prompt: str) -> str:
    t = solve(prompt)
    return f"{t.answer}\n\n[stopped: {t.stopped_because}]\n{t.render()}"
