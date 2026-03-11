"""
LLM-generated MCTS tool: simulation
Description: Improved simulation rollout for ScienceWorld with better action classification, refined keyword boost, and more tolerant stagnation handling.
Generated:   2026-03-11T13:38:48.878352
"""

import random
import re

def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Improved simulation rollout for ScienceWorld.

    Enhancements:
      • Expanded interaction token list and a fallback verb‑object heuristic
        to correctly classify most productive actions.
      • Keyword boost is limited to keywords that also appear among visible objects
        and its multiplier is reduced to avoid spurious attraction.
      • Early‑bonus is granted only when a keyword‑matched action yields an
        immediate score increase.
      • Stagnation detection is more tolerant, allowing neutral interaction/
        navigation steps before cutting off.
    Returns the reward for the requested player.
    """
    # ------------------------------------------------------------------
    # 0) Helper constants & functions
    # ------------------------------------------------------------------
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "if", "with", "to", "of", "in",
        "on", "at", "by", "for", "from", "up", "down", "out", "over", "under",
        "as", "is", "it", "its", "be", "have", "has", "had", "do", "does",
        "did", "was", "were", "am", "are", "not", "no", "this", "that",
        "these", "those", "etc", "into", "onto", "off", "my", "your", "our"
    }

    # Primary token groups (lower‑case substrings)
    INTERACT_TOKENS = (
        "pick up", "pick", "take", "grab", "use", "turn on", "turn off",
        "activate", "open", "close", "put", "place", "pour", "mix",
        "heat", "cool", "boil", "freeze", "melt", "measure", "test",
        "connect", "break", "press", "push", "pull", "cut", "attach",
        "detach", "teleport", "teleport to", "go to", "move to", "walk to"
    )
    NAV_TOKENS = (
        "go ", "move ", "enter ", "walk ", "run ", "teleport", "travel"
    )
    OBS_TOKENS = (
        "look", "inventory", "focus", "examine", "describe", "check", "read"
    )

    # Verb list used for fallback classification (must be lower‑case)
    VERB_TOKENS = (
        "pick", "take", "grab", "use", "open", "close", "activate",
        "boil", "heat", "measure", "mix", "pour", "turn on", "turn off",
        "teleport", "go", "move", "walk"
    )

    # Weights (strongly favor interact/nav over observation so rollouts make progress)
    INTERACT_WEIGHT = 8.0
    NAV_WEIGHT      = 2.0
    OBS_WEIGHT      = 0.02   # very low so "look" is rarely chosen in rollout
    DEFAULT_WEIGHT = 1.0
    KEYWORD_MULT    = 2.0   # reduced to avoid over‑attraction
    OBJECT_BOOST    = 3.0

    # Simple stemmer: strip common suffixes
    def stem(word: str) -> str:
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)]
        return word

    # ------------------------------------------------------------------
    # 1) Extract stemmed keywords from the task description.
    # ------------------------------------------------------------------
    task_text = state.task_description.lower()
    raw_kw = re.findall(r"[a-zA-Z]+", task_text)
    keywords = {
        stem(w) for w in raw_kw if len(w) > 2 and w not in STOP_WORDS
    }

    # ------------------------------------------------------------------
    # 2) Parse visible objects from the current observation.
    # ------------------------------------------------------------------
    obs_text = state.observation_text().lower()
    raw_objs = re.findall(r"[a-zA-Z]+", obs_text)
    visible_objects = {
        stem(w) for w in raw_objs if len(w) > 2 and w not in STOP_WORDS
    }

    # ------------------------------------------------------------------
    # 3) Action classification with fallback verb‑object heuristic.
    # ------------------------------------------------------------------
    def classify_action(action: str) -> str:
        low = action.lower()
        if any(tok in low for tok in INTERACT_TOKENS):
            return "interact"
        if any(tok in low for tok in NAV_TOKENS):
            return "nav"
        if any(tok in low for tok in OBS_TOKENS):
            return "obs"
        # fallback: look for a verb *and* a visible object
        if any(v in low for v in VERB_TOKENS) and any(obj in low for obj in visible_objects):
            return "interact"
        return "other"

    # ------------------------------------------------------------------
    # 4) Roll‑out loop.
    # ------------------------------------------------------------------
    sim_state = state.clone()
    depth = 0
    prev_score = sim_state.normalized_score()
    no_progress_steps = 0          # counts consecutive non‑progress steps
    early_bonus = False

    while not sim_state.is_terminal() and depth < max_depth:
        actions = sim_state.legal_actions()
        if not actions:
            break

        # ----- compute weight for each candidate action -----
        weights = []
        for act in actions:
            act_type = classify_action(act)
            if act_type == "interact":
                w = INTERACT_WEIGHT
            elif act_type == "nav":
                w = NAV_WEIGHT
            elif act_type == "obs":
                w = OBS_WEIGHT
            else:
                w = DEFAULT_WEIGHT

            low = act.lower()

            # Keyword boost – only if keyword also appears among visible objects
            if any(kw in low for kw in keywords):
                if any(obj in low for obj in visible_objects):
                    w *= KEYWORD_MULT
                else:
                    w *= 1.5  # mild boost for pure keyword matches

            # Visible‑object boost
            if any(obj in low for obj in visible_objects):
                w *= OBJECT_BOOST

            weights.append(w)

        # ----- weighted random selection -----
        total_weight = sum(weights)
        if total_weight == 0:
            chosen_action = random.choice(actions)
        else:
            r = random.random() * total_weight
            cum = 0.0
            chosen_action = actions[-1]  # fallback
            for act, w in zip(actions, weights):
                cum += w
                if r <= cum:
                    chosen_action = act
                    break

        # ----- apply chosen action -----
        sim_state.apply_action(chosen_action)
        depth += 1

        current_score = sim_state.normalized_score()
        delta_score = current_score - prev_score

        # Early‑bonus only when a keyword‑matched action yields a score increase
        if delta_score > 0 and any(kw in chosen_action.lower() for kw in keywords):
            early_bonus = True

        # ----- stagnation handling -----
        if delta_score > 0:
            no_progress_steps = 0
            prev_score = current_score
        else:
            act_type = classify_action(chosen_action)
            if act_type == "obs":
                no_progress_steps += 2
            else:
                no_progress_steps += 1

        # Adaptive limit (more forgiving than the original)
        stagnation_limit = max(8, int(depth * 0.3) + 8)
        if no_progress_steps >= stagnation_limit:
            break

        if current_score >= 1.0:
            break

    # ------------------------------------------------------------------
    # 5) Return reward: use shaped reward so short rollouts still differentiate.
    # ------------------------------------------------------------------
    # With short max_depth, rollouts rarely reach terminal state, so returns()[0] is
    # almost always 0 and every action gets the same value. Use normalized_score() as
    # the reward so that actions leading to progress (e.g. interact, go to kitchen)
    # get higher value than pure "look" rollouts that never change score.
    terminal_returns = sim_state.returns()
    terminal_reward = terminal_returns[perspective_player] if terminal_returns else 0.0
    shaped = sim_state.normalized_score()
    # Prefer terminal reward when we actually finished; else use shaped (progress) reward
    reward = terminal_reward if sim_state.is_terminal() else shaped
    if early_bonus:
        reward = min(1.0, reward + 0.1)
    return reward
