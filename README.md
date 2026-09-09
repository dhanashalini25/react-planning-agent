# 03 - ReAct Planning Agent

> Observe, think, act - with a hard stop.

**What it demonstrates:** Building a tool-using loop that provably terminates

**Status:** working implementation with passing tests. Built as a learning project to understand the pattern, not as a production service.

---

## Run it right now

No API key needed - every project ships with `MODEL=fake`, a deterministic
offline responder, so you can see the whole flow work before spending anything.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
python -m src.main
pytest -q
```

To use a real model, edit `.env`:

```
MODEL=gpt-4o-mini            # + OPENAI_API_KEY
MODEL=claude-3-5-haiku-latest  # + ANTHROPIC_API_KEY
MODEL=ollama/llama3.1        # free, runs locally
```

## How it works

The classic ReAct loop, written as explicit state rather than `while True`. Each turn parses `Thought / Action / Action Input` (or `Final Answer`) out of the reply, runs the named tool, appends the observation to a scratchpad, and goes round again.

Three things stop it: a step cap, a wall-clock cap, and loop detection - if the model asks for the same tool with the same argument twice in a row, the run is cut short. In every case `degrade()` returns a partial answer built from what was actually observed, so the caller gets something useful rather than an exception.

The calculator uses an AST walk rather than `eval`, so a prompt-injected expression can't run arbitrary code.

## What "done" means here

- The loop is explicit state with a step cap and a wall-clock cap
- Repeating the same tool call twice in a row breaks the loop
- Stopping early returns a partial answer, never an exception
- A full printable trace of thoughts, actions and observations ships with the answer
- The calculator tool cannot execute arbitrary Python
- A deliberately looping model is in the test suite and the agent terminates

Every one of those lines has a test behind it in `tests/` - `pytest -q` is the
proof, not the README.

## Layout

```
src/llm.py             provider-agnostic completion, plus offline fake mode
src/fake.py            the canned responses that make MODEL=fake work
src/logging_setup.py   structured JSON logging
src/agent.py           the pattern itself
src/main.py            CLI entrypoint
tests/                 11 tests, all passing
```

## Next steps

- Port the loop to LangGraph and keep the same tests green
- Add a reflection step after a failed tool call instead of stopping immediately
- Measure how often loop detection fires on a real model versus the fake one

## Reference

https://github.com/langchain-ai/react-agent

---

Part of a 12-project agentic AI series - [github.com/dhanashalini25](https://github.com/dhanashalini25?tab=repositories)
