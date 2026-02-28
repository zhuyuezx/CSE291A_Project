def test_transfer_eval_smoke():
    """Transfer evaluation should run without errors on a tiny budget."""
    from experiments.transfer_eval import run_transfer_chain
    results = run_transfer_chain(
        source_game="connect_four",
        target_game="pathfinding",
        source_tool_dir="tool_pool/connect_four",
        target_tool_dir="tool_pool/pathfinding",
        n_eval_games=5,
        sim_budget=20,
    )
    assert "cold_start" in dir(results)
    assert "transferred" in dir(results)
    assert isinstance(results.cold_start.normalized_value, float)
