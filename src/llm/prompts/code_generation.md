# Tool Code Generation

Write a Python tool file implementing the following heuristic for a game-playing MCTS agent.

## Tool Specification
{tool_spec}

## Required Format

The file MUST follow this exact structure:

```python
__TOOL_META__ = {{
    "name": "{tool_name}",
    "type": "{tool_type}",
    "description": "{tool_description}",
}}

def run(state{extra_params}) -> {return_type}:
    """
    {tool_description}

    Args:
        state: An OpenSpiel game state object with methods:
            - state.legal_actions() -> list[int]
            - state.clone() -> State
            - state.apply_action(action: int) -> None (mutates in place)
            - state.is_terminal() -> bool
            - state.returns() -> list[float]
            - state.current_player() -> int
            - str(state) -> str (human-readable board)

    Returns:
        {return_description}
    """
    # Implementation here
```

## Tool Type Signatures
- state_evaluator: `run(state) -> float` (range [-1, 1], positive = good for current player)
- action_filter: `run(state, legal_actions: list[int]) -> list[int]` (subset of legal_actions)
- rollout_policy: `run(state, legal_actions: list[int]) -> int` (single action from legal_actions)
- selection_prior: `run(state, legal_actions: list[int]) -> dict[int, float]` (action -> prior probability)
- reward_shaper: `run(state, raw_value: float) -> float` (shaped reward)
- macro_action: `run(state) -> list[int]` (sequence of primitive actions)

## Rules
- GAME-AGNOSTIC: Do NOT import game-specific modules. Do NOT hardcode board sizes.
- FAST: This runs thousands of times per search. Avoid unnecessary cloning or deep loops.
- SAFE: Handle edge cases (empty action lists, terminal states). Never raise exceptions.
- Use only standard library imports (math, random, collections, etc.)

Output ONLY the Python code, no markdown fences or explanation.
