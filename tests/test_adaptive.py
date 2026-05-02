from mtplx.adaptive import AdaptiveDepthPolicy, ExpectedValueDepthPolicy


def test_adaptive_policy_increases_after_full_accept_streak():
    policy = AdaptiveDepthPolicy(max_depth=4, start_depth=2, increase_after=2)

    first = policy.observe(attempted_depth=2, accepted_depths=2)
    second = policy.observe(attempted_depth=2, accepted_depths=2)

    assert first["action"] == "hold"
    assert second["action"] == "increase"
    assert second["next_depth"] == 3


def test_adaptive_policy_decreases_on_early_reject():
    policy = AdaptiveDepthPolicy(max_depth=5, start_depth=4, decrease_after=1)

    decision = policy.observe(attempted_depth=4, accepted_depths=0)

    assert decision["action"] == "decrease"
    assert decision["next_depth"] == 3


def test_adaptive_policy_clamps_start_depth():
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=2, start_depth=9)

    assert policy.current_depth == 3


def test_adaptive_policy_holds_on_late_reject():
    policy = AdaptiveDepthPolicy(max_depth=5, start_depth=4, decrease_after=1)

    decision = policy.observe(attempted_depth=4, accepted_depths=3)

    assert decision["action"] == "hold"
    assert decision["next_depth"] == 4


def test_expected_value_policy_stops_d3_when_ev_fails():
    policy = ExpectedValueDepthPolicy(
        max_depth=3,
        base_depth=2,
        accept_priors=(0.92, 0.64, 0.32),
        draft_cost_s=0.0048,
        extra_verify_cost_s=0.006,
        baseline_tok_s=40.0,
        safety_margin=0.1,
    )

    decision = policy.should_continue_after_draft(
        drafted_depth=2,
        max_depth=3,
        draft_metrics={"top2_margin": 0.5, "top1_prob_topk": 0.35},
    )

    assert decision["action"] == "stop"
    assert decision["reason"] == "ev_fail"
    assert decision["expected_extra_accept"] < decision["required_extra_accept"]


def test_expected_value_policy_allows_d3_when_ev_clears_cost():
    policy = ExpectedValueDepthPolicy(
        max_depth=3,
        base_depth=2,
        accept_priors=(0.99, 0.96, 0.9),
        draft_cost_s=0.002,
        extra_verify_cost_s=0.002,
        baseline_tok_s=40.0,
        safety_margin=0.05,
    )

    decision = policy.should_continue_after_draft(
        drafted_depth=2,
        max_depth=3,
        draft_metrics={"top2_margin": 6.0, "top1_prob_topk": 0.9},
    )

    assert decision["action"] == "continue"
    assert decision["reason"] == "ev_pass"


def test_expected_value_policy_updates_acceptance_ewma():
    policy = ExpectedValueDepthPolicy(max_depth=3, accept_priors=(0.5, 0.5, 0.5), ewma_alpha=0.5)

    decision = policy.observe(attempted_depth=2, accepted_depths=1)

    assert decision["kind"] == "expected_value"
    assert decision["accept_ewma"][:2] == [0.75, 0.25]
