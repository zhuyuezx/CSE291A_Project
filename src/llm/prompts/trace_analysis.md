# Game Trace Analysis

You are an expert game AI designer. Analyze the following game traces where the agent LOST or performed poorly. Identify patterns and propose a heuristic tool that could help the agent play better.

## Game Description
{game_description}

## Current Tools
{current_tools}

## Game Traces (Losses/Poor Performance)
{traces}

## Task
1. Identify WHY the agent is losing (strategic mistakes, missed opportunities, poor move selection)
2. Propose ONE heuristic tool that would address the most critical weakness
3. Specify the tool as JSON:

```json
{{
  "name": "snake_case_name",
  "type": "state_evaluator|action_filter|rollout_policy|selection_prior|reward_shaper|macro_action",
  "description": "What this tool does and why it helps",
  "pseudocode": "Step-by-step algorithm description"
}}
```

IMPORTANT:
- The tool must be GAME-AGNOSTIC. Use only the OpenSpiel generic API (state.legal_actions(), state.clone(), state.apply_action(), state.is_terminal(), state.returns(), state.current_player(), str(state)).
- Do NOT hardcode board dimensions, piece types, or game-specific rules.
- The tool should be simple and fast (will be called thousands of times per MCTS search).
