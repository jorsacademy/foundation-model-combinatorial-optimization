import unittest

import torch

from rl4co_cvrp_pomo_attention import validate_cvrp_action_sequence


class IndependentCVRPTests(unittest.TestCase):
    def test_valid_sequence(self):
        demand = torch.tensor([[0.4, 0.3, 0.5]], dtype=torch.float32)
        capacity = torch.tensor([[1.0]])
        actions = torch.tensor([[1, 2, 0, 3, 0]], dtype=torch.long)
        self.assertTrue(validate_cvrp_action_sequence(actions, demand, capacity))

    def test_duplicate_customer_rejected(self):
        demand = torch.tensor([[0.4, 0.3, 0.5]], dtype=torch.float32)
        capacity = torch.tensor([[1.0]])
        actions = torch.tensor([[1, 2, 0, 2, 0]], dtype=torch.long)
        self.assertFalse(validate_cvrp_action_sequence(actions, demand, capacity))

    def test_capacity_overload_rejected(self):
        demand = torch.tensor([[0.4, 0.3, 0.5]], dtype=torch.float32)
        capacity = torch.tensor([[1.0]])
        actions = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
        self.assertFalse(validate_cvrp_action_sequence(actions, demand, capacity))


if __name__ == "__main__":
    unittest.main()
