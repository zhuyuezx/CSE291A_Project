# src/tools/generator.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from src.llm.client import LLMClient
from src.tools.validator import ToolValidator, ValidationResult


@dataclass
class GenerationResult:
    valid: bool
    code: str | None = None
    spec: dict | None = None
    error: str | None = None


PROMPTS_DIR = Path(__file__).parent.parent / "llm" / "prompts"


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    with open(path) as f:
        return f.read()


class ToolGenerator:
    def __init__(
        self,
        trace_analyzer_client: LLMClient,
        code_generator_client: LLMClient,
        validator_client: LLMClient,
        game_name: str = "tic_tac_toe",
        max_retries: int = 3,
    ):
        self.trace_analyzer = trace_analyzer_client
        self.code_generator = code_generator_client
        self.validator_client = validator_client
        self.game_name = game_name
        self.max_retries = max_retries
        self.tool_validator = ToolValidator(game_name=game_name)

    def analyze_traces(
        self,
        traces_text: str,
        game_description: str,
        current_tools_desc: str,
    ) -> dict | None:
        print("  [ToolGen] Phase: TRACE ANALYSIS")
        print(f"  [ToolGen]   Traces input: {len(traces_text)} chars")
        print(f"  [ToolGen]   Game: {game_description}")
        print(f"  [ToolGen]   Current tools: {current_tools_desc}")
        t0 = time.time()

        try:
            prompt_template = _load_prompt("trace_analysis.md")
        except FileNotFoundError:
            prompt_template = (
                "Analyze these game traces and propose a heuristic tool.\n"
                "Game: {game_description}\n"
                "Current tools: {current_tools}\n"
                "Traces:\n{traces}\n"
                "Respond with JSON: {name, type, description, pseudocode}"
            )

        prompt = prompt_template.format(
            game_description=game_description,
            current_tools=current_tools_desc,
            traces=traces_text,
        )

        response = self.trace_analyzer.generate(
            system_prompt="You are a game AI expert. Respond with valid JSON only.",
            user_prompt=prompt,
        )

        # Parse JSON from response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                spec = json.loads(response[start:end])
                elapsed = time.time() - t0
                print(f"  [ToolGen]   ✓ Analysis complete in {elapsed:.1f}s")
                print(f"  [ToolGen]   Proposed: name={spec.get('name')}, type={spec.get('type')}")
                print(f"  [ToolGen]   Description: {spec.get('description', '')[:120]}")
                return spec
        except json.JSONDecodeError as e:
            print(f"  [ToolGen]   ✗ JSON parse error: {e}")
            print(f"  [ToolGen]   Raw response (first 300 chars): {response[:300]}")
        return None

    def generate_code(self, spec: dict) -> str:
        print("  [ToolGen] Phase: CODE GENERATION")
        print(f"  [ToolGen]   Generating: {spec['name']} ({spec['type']})")
        t0 = time.time()

        tool_type = spec["type"]
        extra_params = ""
        return_type = "float"
        return_desc = "float value"

        type_signatures = {
            "state_evaluator": ("", "float", "Score in [-1, 1]"),
            "action_filter": (
                ", legal_actions: list[int]",
                "list[int]",
                "Subset of legal_actions",
            ),
            "rollout_policy": (
                ", legal_actions: list[int]",
                "int",
                "Single action from legal_actions",
            ),
            "selection_prior": (
                ", legal_actions: list[int]",
                "dict[int, float]",
                "Action to prior probability mapping",
            ),
            "reward_shaper": (
                ", raw_value: float",
                "float",
                "Shaped reward value",
            ),
            "macro_action": ("", "list[int]", "Sequence of primitive actions"),
        }

        if tool_type in type_signatures:
            extra_params, return_type, return_desc = type_signatures[tool_type]

        try:
            prompt_template = _load_prompt("code_generation.md")
        except FileNotFoundError:
            prompt_template = (
                "Write a Python tool implementing: {tool_spec}\n"
                "Name: {tool_name}, Type: {tool_type}\n"
                "Function signature: def run(state{extra_params}) -> {return_type}\n"
                "Output only Python code."
            )

        prompt = prompt_template.format(
            tool_spec=json.dumps(spec, indent=2),
            tool_name=spec["name"],
            tool_type=tool_type,
            tool_description=spec["description"],
            extra_params=extra_params,
            return_type=return_type,
            return_description=return_desc,
        )

        response = self.code_generator.generate(
            system_prompt="You are a Python programmer. Output only valid Python code, no markdown.",
            user_prompt=prompt,
        )

        # Strip markdown fences if present
        code = response.strip()
        if code.startswith("```python"):
            code = code[len("```python") :].strip()
        if code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()

        elapsed = time.time() - t0
        print(f"  [ToolGen]   ✓ Code generated in {elapsed:.1f}s ({len(code)} chars, {code.count(chr(10))+1} lines)")

        return code

    def generate_tool(
        self,
        traces_text: str,
        game_description: str,
        current_tools_desc: str,
    ) -> GenerationResult:
        print("=" * 60)
        print("  [ToolGen] STARTING TOOL GENERATION PIPELINE")
        print("=" * 60)
        pipeline_t0 = time.time()

        # Step 1: Analyze traces
        spec = self.analyze_traces(traces_text, game_description, current_tools_desc)
        if spec is None:
            print("  [ToolGen] ✗ PIPELINE FAILED: Could not analyze traces")
            return GenerationResult(valid=False, error="Failed to analyze traces")

        # Validate spec has required keys
        if "name" not in spec:
            spec["name"] = "llm_generated_tool"
            print("  [ToolGen]   ⚠ No 'name' in spec, defaulting to 'llm_generated_tool'")
        if "type" not in spec:
            spec["type"] = "state_evaluator"
            print("  [ToolGen]   ⚠ No 'type' in spec, defaulting to 'state_evaluator'")
        if "description" not in spec:
            spec["description"] = "LLM-generated heuristic tool"
            print("  [ToolGen]   ⚠ No 'description' in spec, defaulting")

        # Step 2: Generate code
        code = self.generate_code(spec)

        # Step 3: Validate (with retries)
        print("  [ToolGen] Phase: VALIDATION")
        for attempt in range(self.max_retries):
            print(f"  [ToolGen]   Attempt {attempt + 1}/{self.max_retries}...")
            t0 = time.time()
            result = self.tool_validator.validate_code(code)
            elapsed = time.time() - t0
            if result.valid:
                total = time.time() - pipeline_t0
                print(f"  [ToolGen]   ✓ Validation PASSED in {elapsed:.1f}s")
                print(f"  [ToolGen] ✓ PIPELINE COMPLETE in {total:.1f}s total")
                print("=" * 60)
                return GenerationResult(valid=True, code=code, spec=spec)

            print(f"  [ToolGen]   ✗ Validation FAILED in {elapsed:.1f}s: {result.error}")

            # Try to fix with LLM
            print(f"  [ToolGen]   Requesting LLM fix...")
            try:
                fix_prompt = (
                    f"Fix this tool code. Error: {result.error}\n\n"
                    f"Original code:\n{code}\n\n"
                    "Output only the fixed Python code."
                )
                code = self.validator_client.generate(
                    system_prompt="You are a Python debugger. Output only valid Python code.",
                    user_prompt=fix_prompt,
                )
                code = code.strip()
                if code.startswith("```python"):
                    code = code[len("```python") :].strip()
                if code.endswith("```"):
                    code = code[:-3].strip()
                print(f"  [ToolGen]   Received fixed code ({len(code)} chars)")
            except Exception as e:
                print(f"  [ToolGen]   ✗ LLM fix failed: {e}")

        total = time.time() - pipeline_t0
        print(f"  [ToolGen] ✗ PIPELINE FAILED after {self.max_retries} retries ({total:.1f}s)")
        print("=" * 60)
        return GenerationResult(
            valid=False,
            code=code,
            spec=spec,
            error=f"Failed validation after {self.max_retries} retries",
        )

