"""
Multi-API Batch LLM Template
==============================
Process many inputs in parallel using multiple API keys.

This template:
  1. Loads a batch of inputs from a JSON file
  2. Distributes them across multiple API keys (round-robin)
  3. Sends all requests concurrently via asyncio (no MPI needed!)
  4. Optionally runs MCP tool calls in the loop
  5. Saves all results to a JSON output file

Dependencies:
    pip install openai python-dotenv

Usage:
    # 1. Edit the CONFIG section below (API keys, model, base URL)
    # 2. Create an input JSON file (see example below or run --example-input)
    # 3. Run:
    python multi_mcp_llm_template.py --input inputs.json --output results.json

    # Generate an example input file:
    python multi_mcp_llm_template.py --example-input

    # Control concurrency (default = number of API keys):
    python multi_mcp_llm_template.py --input inputs.json --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    # Walk up to find .env in Tool_Creation/
    _ENV_PATH = Path(__file__).resolve().parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
    else:
        load_dotenv()
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# ── Multiple API keys ─────────────────────────────────────────────────────────
# Add as many as you have. Requests are distributed round-robin across them.
try:
    API_KEYS: list[str] = os.getenv("API_KEYS").split(",")
except Exception:
    print("Error: API_KEYS environment variable not set or invalid. "
          "Set it to a comma-separated list of keys (e.g. 'key1,key2,key3').")

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://tritonai-api.ucsd.edu")
MODEL = os.getenv("MODEL_NAME", "api-gpt-oss-120b")

# ── Default system prompt ─────────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely and accurately. "
    "You have access to tools you can call when needed."
)

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "batch_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Tool Definitions (same as before, but async-compatible)
# ══════════════════════════════════════════════════════════════════════════════
# Define your tools here. These run locally — the LLM just *requests* them.

def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(ch in allowed for ch in expression):
        return f"Error: disallowed characters in expression."
    try:
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"


def get_current_time(timezone_offset: int = 0) -> str:
    """Return current time with optional timezone offset in hours."""
    from datetime import timedelta
    now = datetime.now(timezone.utc) + timedelta(hours=timezone_offset)
    return now.strftime("%Y-%m-%d %H:%M:%S")


def search_knowledge_base(query: str) -> str:
    """Search a local knowledge base (stub — replace with your logic)."""
    knowledge = {
        "python": "Python is a high-level programming language.",
        "mcp": "MCP standardizes how LLMs connect to external tools.",
        "openai": "OpenAI provides API access to language models.",
    }
    results = [v for k, v in knowledge.items() if k in query.lower()]
    return "\n".join(results) if results else f"No results for '{query}'."


# ── Tool registry ────────────────────────────────────────────────────────────

TOOL_DISPATCH: dict[str, callable] = {
    "calculate": calculate,
    "get_current_time": get_current_time,
    "search_knowledge_base": search_knowledge_base,
}

OPENAI_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression. E.g. '2 + 3', '10 * (4-1)'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current time with optional timezone offset in hours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone_offset": {
                        "type": "integer",
                        "description": "Hours from UTC (e.g. -8 for PST)",
                        "default": 0,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search a local knowledge base for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name and return its string result."""
    func = TOOL_DISPATCH.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'"
    try:
        return str(func(**arguments))
    except Exception as e:
        return f"Error in tool '{name}': {e}"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Input / Output Format
# ══════════════════════════════════════════════════════════════════════════════
#
# Input JSON file format:
# {
#     "system_prompt": "You are a helpful assistant.",       ← optional, override default
#     "use_tools": true,                                     ← optional, default true
#     "inputs": [
#         {
#             "id": "q1",                                    ← unique identifier
#             "message": "What is 42 * 17?",                 ← the text prompt
#             "data": {"key": "value"}                       ← optional structured data
#         },
#         {
#             "id": "q2",
#             "message": "Summarize this data",
#             "data": {"users": 1500, "revenue": 25000}
#         }
#     ]
# }
#
# Output JSON file format:
# {
#     "metadata": {
#         "model": "api-gpt-oss-120b",
#         "total_inputs": 10,
#         "successful": 9,
#         "failed": 1,
#         "elapsed_seconds": 12.5
#     },
#     "results": [
#         {
#             "id": "q1",
#             "input_message": "What is 42 * 17?",
#             "input_data": null,
#             "response": "42 * 17 = 714",
#             "tool_calls": [{"name": "calculate", "args": {"expression": "42*17"}, "result": "714"}],
#             "status": "success",
#             "api_key_index": 0,
#             "elapsed_seconds": 2.1
#         },
#         ...
#     ]
# }

EXAMPLE_INPUT = {
    "system_prompt": "You are a helpful assistant that can use tools when needed.",
    "use_tools": True,
    "inputs": [
        {
            "id": "math_1",
            "message": "What is 123 * 456?",
        },
        {
            "id": "math_2",
            "message": "Calculate 2^10 + 3^5",
        },
        {
            "id": "time_1",
            "message": "What time is it in PST (UTC-8)?",
        },
        {
            "id": "knowledge_1",
            "message": "What do you know about Python?",
        },
        {
            "id": "data_analysis",
            "message": "Analyze this sales data and identify the top performer.",
            "data": {
                "sales": [
                    {"name": "Alice", "amount": 15000},
                    {"name": "Bob", "amount": 22000},
                    {"name": "Carol", "amount": 18500},
                ]
            },
        },
        {
            "id": "no_tool_needed",
            "message": "Explain what MCP (Model Context Protocol) is in one sentence.",
        },

        # ── Code modification examples ────────────────────────────────
        # Put the code as a string in "data.code", and describe the
        # modification you want in "message".
        {
            "id": "code_modify_1",
            "message": "Add type hints to this function and add a docstring. Return ONLY the modified code.",
            "data": {
                "code": (
                    "def process_items(items, threshold):\n"
                    "    results = []\n"
                    "    for item in items:\n"
                    "        if item['score'] > threshold:\n"
                    "            results.append(item['name'])\n"
                    "    return results\n"
                ),
                "language": "python",
            },
        },
        {
            "id": "code_modify_2",
            "message": (
                "Refactor this class: extract the validation logic into a separate method, "
                "and add error handling. Return ONLY the modified code."
            ),
            "data": {
                "code": (
                    "class UserManager:\n"
                    "    def __init__(self):\n"
                    "        self.users = []\n"
                    "\n"
                    "    def add_user(self, name, email, age):\n"
                    "        if not name or len(name) < 2:\n"
                    "            return False\n"
                    "        if '@' not in email:\n"
                    "            return False\n"
                    "        if age < 0 or age > 150:\n"
                    "            return False\n"
                    "        self.users.append({'name': name, 'email': email, 'age': age})\n"
                    "        return True\n"
                ),
                "language": "python",
            },
        },
        {
            "id": "code_modify_3",
            "message": "Convert this synchronous code to async. Return ONLY the modified code.",
            "data": {
                "code": (
                    "import requests\n"
                    "\n"
                    "def fetch_all(urls):\n"
                    "    results = []\n"
                    "    for url in urls:\n"
                    "        resp = requests.get(url)\n"
                    "        results.append(resp.json())\n"
                    "    return results\n"
                ),
                "language": "python",
            },
        },
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Async Batch Processor
# ══════════════════════════════════════════════════════════════════════════════
# Uses asyncio to send all requests concurrently. No MPI needed — LLM calls
# are I/O-bound (waiting for network), so async is the right approach.
# A semaphore limits how many requests are in-flight at once.

from openai import AsyncOpenAI


async def process_single_input(
    item: dict,
    client: AsyncOpenAI,
    api_key_index: int,
    system_prompt: str,
    use_tools: bool,
    semaphore: asyncio.Semaphore,
    max_tool_rounds: int = 5,
) -> dict:
    """
    Process one input item: send to LLM, handle tool calls, return result.

    The semaphore limits concurrency so you don't overwhelm the API.
    """
    async with semaphore:
        item_id = item.get("id", "unknown")
        message_text = item.get("message", "")
        input_data = item.get("data", None)

        # ── Build the user message ────────────────────────────────────────
        # If there's structured data, include it in the message as JSON
        if input_data is not None:
            user_content = (
                f"{message_text}\n\n"
                f"Here is the relevant data:\n"
                f"```json\n{json.dumps(input_data, indent=2)}\n```"
            )
        else:
            user_content = message_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        tool_calls_log: list[dict] = []
        start_time = time.time()

        try:
            for _ in range(max_tool_rounds):
                kwargs: dict[str, Any] = {
                    "model": MODEL,
                    "messages": messages,
                }
                if use_tools and OPENAI_TOOLS:
                    kwargs["tools"] = OPENAI_TOOLS
                    kwargs["tool_choice"] = "auto"

                response = await client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                assistant_msg = choice.message

                # Build assistant message for history
                msg_dict: dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                }

                if assistant_msg.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_msg.tool_calls
                    ]
                    messages.append(msg_dict)

                    # Execute tools locally
                    for tc in assistant_msg.tool_calls:
                        fn_name = tc.function.name
                        fn_args = json.loads(tc.function.arguments)
                        result = execute_tool(fn_name, fn_args)
                        tool_calls_log.append({
                            "name": fn_name,
                            "args": fn_args,
                            "result": result,
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                    continue  # loop back for LLM to incorporate tool results

                # No tool calls → final answer
                messages.append(msg_dict)
                elapsed = time.time() - start_time
                return {
                    "id": item_id,
                    "input_message": message_text,
                    "input_data": input_data,
                    "response": assistant_msg.content or "",
                    "tool_calls": tool_calls_log,
                    "full_conversation": messages,
                    "status": "success",
                    "api_key_index": api_key_index,
                    "elapsed_seconds": round(elapsed, 2),
                }

            # Exhausted tool rounds
            elapsed = time.time() - start_time
            return {
                "id": item_id,
                "input_message": message_text,
                "input_data": input_data,
                "response": messages[-1].get("content", ""),
                "tool_calls": tool_calls_log,
                "full_conversation": messages,
                "status": "max_tool_rounds_reached",
                "api_key_index": api_key_index,
                "elapsed_seconds": round(elapsed, 2),
            }

        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "id": item_id,
                "input_message": message_text,
                "input_data": input_data,
                "response": None,
                "tool_calls": tool_calls_log,
                "full_conversation": messages,
                "status": "error",
                "error": str(e),
                "api_key_index": api_key_index,
                "elapsed_seconds": round(elapsed, 2),
            }


async def run_batch(
    inputs: list[dict],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    use_tools: bool = True,
    concurrency: int | None = None,
) -> list[dict]:
    """
    Process all inputs in parallel, distributing across API keys.

    Args:
        inputs:      List of input dicts (each must have 'id' and 'message').
        system_prompt: System prompt for all requests.
        use_tools:   Whether to enable tool calling.
        concurrency: Max concurrent requests (defaults to len(API_KEYS)).

    Returns:
        List of result dicts.
    """
    if not API_KEYS or API_KEYS == [""]:
        raise ValueError("No API keys configured! Set API_KEYS env var or edit the CONFIG section.")

    max_concurrent = concurrency or len(API_KEYS)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create one AsyncOpenAI client per API key
    clients = [
        AsyncOpenAI(api_key=key.strip(), base_url=BASE_URL)
        for key in API_KEYS
    ]

    # Round-robin assign each input to a client
    key_cycle = cycle(range(len(clients)))

    tasks = []
    for item in inputs:
        key_idx = next(key_cycle)
        task = process_single_input(
            item=item,
            client=clients[key_idx],
            api_key_index=key_idx,
            system_prompt=system_prompt,
            use_tools=use_tools,
            semaphore=semaphore,
        )
        tasks.append(task)

    # Fire all requests concurrently and wait for all to finish
    print(f"🚀 Sending {len(tasks)} requests (concurrency={max_concurrent}, "
          f"api_keys={len(clients)}, tools={'on' if use_tools else 'off'})")

    results = await asyncio.gather(*tasks)
    return list(results)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLI & Main
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results: list[dict], output_path: Path, elapsed: float) -> None:
    """Save results to a JSON file with metadata."""
    successful = sum(1 for r in results if r["status"] == "success")
    output = {
        "metadata": {
            "model": MODEL,
            "base_url": BASE_URL,
            "num_api_keys": len(API_KEYS),
            "total_inputs": len(results),
            "successful": successful,
            "failed": len(results) - successful,
            "total_elapsed_seconds": round(elapsed, 2),
        },
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved → {output_path}")


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point."""
    # Load inputs
    with open(args.input, "r") as f:
        input_data = json.load(f)

    system_prompt = input_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    use_tools = input_data.get("use_tools", True)
    inputs = input_data["inputs"]

    print(f"📋 Loaded {len(inputs)} inputs from {args.input}")

    # Run batch
    start = time.time()
    results = await run_batch(
        inputs=inputs,
        system_prompt=system_prompt,
        use_tools=use_tools,
        concurrency=args.concurrency,
    )
    elapsed = time.time() - start

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ✅ Done in {elapsed:.1f}s")
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        tools_used = ", ".join(tc["name"] for tc in r.get("tool_calls", []))
        tools_str = f" [tools: {tools_used}]" if tools_used else ""
        response_preview = (r.get("response") or "(no response)")[:80]
        print(f"  {status_icon} {r['id']}: {response_preview}{tools_str}")
    print(f"{'='*60}\n")

    # Save
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"results_{int(time.time())}.json"
    save_results(results, output_path, elapsed)


def main():
    parser = argparse.ArgumentParser(
        description="Batch LLM processing with multiple API keys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate example input file:
  python multi_mcp_llm_template.py --example-input

  # Process a batch:
  python multi_mcp_llm_template.py --input inputs.json --output results.json

  # With higher concurrency:
  python multi_mcp_llm_template.py --input inputs.json --concurrency 10
        """,
    )
    parser.add_argument("--input", type=str, help="Path to input JSON file")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON file")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Max concurrent requests (default: number of API keys)")
    parser.add_argument("--example-input", action="store_true",
                        help="Generate an example input JSON file and exit")

    args = parser.parse_args()

    if args.example_input:
        out_path = Path("inputs_example.json")
        with open(out_path, "w") as f:
            json.dump(EXAMPLE_INPUT, f, indent=2)
        print(f"📄 Example input written to {out_path}")
        print(f"   Edit it, then run: python {sys.argv[0]} --input {out_path}")
        return

    if not args.input:
        parser.error("--input is required (or use --example-input to generate an example)")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
