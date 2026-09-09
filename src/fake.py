"""Canned ReAct transcript for MODEL=fake: one tool call, then an answer."""
from __future__ import annotations


def respond(messages: list[dict]) -> str:
    scratch = messages[-1]["content"]
    if "Observation:" not in scratch:
        return (
            "Thought: I should compute the product first.\n"
            "Action: calculator\n"
            "Action Input: 128 * 47"
        )
    if scratch.count("Observation:") == 1:
        return (
            "Thought: Now I need the knowledge base entry.\n"
            "Action: knowledge_base\n"
            "Action Input: retries"
        )
    return (
        "Thought: I have both pieces.\n"
        "Final Answer: 128 * 47 is 6016, and the agent retries up to three times."
    )
