"""Policy engine tests."""

def test_expected_costs():
    from ring_sentinel.policy.engine import expected_costs
    from ring_sentinel.config import load_config
    
    cfg = load_config()
    costs = cfg.costs
    
    # Test a fraud case (high score)
    c_a, c_b, c_r = expected_costs(0.9, 10000.0, costs)
    assert c_a > c_b, "blocking should be cheaper than allowing fraud"
    
    # Test a legit case (low score)
    c_a, c_b, c_r = expected_costs(0.1, 10000.0, costs)
    assert c_b > c_a, "allowing legit should be cheaper than blocking"
