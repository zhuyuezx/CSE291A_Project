"""
LLM-generated MCTS tool: simulation
Description: 
Generated:   2026-03-11T15:39:40.814477
"""

import random


def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Heuristic‑guided simulation for Zork.

    The rollout is no longer pure random; actions receive a
    lightweight score based on game‑specific knowledge:
        • pick up treasures,
        • deposit them in the trophy case,
        • manage the lantern (turn on/off, respect fuel),
        • attack the troll when possible,
        • discourage useless information actions,
        • avoid immediate repetitions.
    The function keeps the original signature so it can be used
    directly by the MCTS pipeline.
    """
    sim_state = state.clone()
    depth = 0
    recent_actions: list[str] = []          # for simple loop detection

    while not sim_state.is_terminal() and depth < max_depth:
        # Adjust depth limit according to remaining lantern fuel (if it is on)
        if getattr(sim_state, "lantern_on", False):
            fuel_limit = sim_state.lantern_fuel + 10
            if depth >= fuel_limit:
                break

        # ------------------------------------------------------------------
        # Goal shortcut: if we already carry a treasure and we are in the
        # living room, force a deposit action and skip the normal weighting.
        # ------------------------------------------------------------------
        treasures_held = [
            itm for itm in sim_state.inventory
            if itm in sim_state.treasures_remaining()
        ]
        if treasures_held and sim_state.room == "living room":
            # Choose the first depositable treasure we can put down.
            for t in treasures_held:
                put_action = f"put {t} in trophy case"
                if put_action in sim_state.legal_actions():
                    sim_state.apply_action(put_action)
                    depth += 1
                    # after depositing we continue the rollout normally
                    break
            continue

        actions = sim_state.legal_actions()
        weights: list[float] = []

        # Pre‑compute some simple flags
        has_sword = any("sword" in itm.lower() for itm in sim_state.inventory)

        for a in actions:
            w = 1.0  # baseline weight

            # ----- Treasure handling -------------------------------------------------
            if a.startswith("take "):
                item = a[5:]
                if item in sim_state.treasures_remaining():
                    w += 3.0

            if a.startswith("put ") and "trophy case" in a and sim_state.room == "living room":
                # only reward putting a treasure, not any other object
                parts = a.split()
                if len(parts) >= 2:
                    item = parts[1]
                    if item in sim_state.treasures_remaining():
                        w += 5.0

            # ----- Lantern management ------------------------------------------------
            if a == "turn on lantern":
                if not getattr(sim_state, "lantern_on", False):
                    w += 2.0

            if a == "turn off lantern":
                if getattr(sim_state, "lantern_on", False):
                    # strongly encourage turning off when fuel is low
                    if getattr(sim_state, "lantern_fuel", 0) < 15:
                        w += 3.0
                    else:
                        w += 1.0

            # ----- Troll handling ----------------------------------------------------
            if "attack troll" in a:
                if getattr(sim_state, "troll_alive", True) and has_sword and getattr(sim_state, "lantern_on", False):
                    w += 4.0

            # ----- Movement -----------------------------------------------------------
            if a.startswith("go "):
                # small incentive to move; we cannot compute exact distance to treasures
                # without a mapping room‑>treasure, so we keep it modest.
                w += 0.5

                # simple safety: if lantern is off, avoid moving underground (unknown)
                # (no direct info, so we do not penalise here)

            # ----- Information actions ------------------------------------------------
            if a in ("look", "inventory"):
                w -= 2.0   # penalise as they give no progress in punishment mode

            # ----- Loop detection -----------------------------------------------------
            if recent_actions and a == recent_actions[-1]:
                w = 0.0   # break immediate repetitions (e.g., look → look)

            # Ensure non‑negative weight for random.choices
            if w < 0:
                w = 0.0

            weights.append(w)

        # If every weight became zero (unlikely), fall back to uniform sampling.
        if all(w == 0 for w in weights):
            weights = [1.0] * len(actions)

        # Choose action proportionally to its weight.
        chosen_action = random.choices(actions, weights=weights, k=1)[0]
        sim_state.apply_action(chosen_action)

        # Update recent‑action history
        recent_actions.append(chosen_action)
        if len(recent_actions) > 3:
            recent_actions.pop(0)

        depth += 1

    # Return the reward for the perspective player (root player).
    return sim_state.returns()[perspective_player]
