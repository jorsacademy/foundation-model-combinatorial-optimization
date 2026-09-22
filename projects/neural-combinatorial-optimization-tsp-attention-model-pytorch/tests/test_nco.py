import math
import unittest

import numpy as np
import torch

from nco_tsp import (
    AttentionModel,
    GreedyRolloutBaseline,
    ModelConfig,
    brute_force_tsp,
    generate_tsp_batch,
    held_karp_tsp,
    nearest_neighbor_tour,
    reinforce_train_step,
    tour_length,
    validate_tour,
)


class NeuralCombinatorialOptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.config = ModelConfig(
            embedding_dim=32,
            n_heads=4,
            n_encoder_layers=2,
            feed_forward_dim=64,
        )
        self.model = AttentionModel(self.config)

    def test_encoder_shape_and_gradient(self):
        coords = generate_tsp_batch(4, 7).requires_grad_(True)
        encoded = self.model.encoder(coords)
        self.assertEqual(encoded.shape, (4, 7, 32))
        encoded.sum().backward()
        self.assertIsNotNone(coords.grad)
        self.assertTrue(torch.isfinite(coords.grad).all())

    def test_sample_decoder_returns_valid_permutations(self):
        coords = generate_tsp_batch(16, 9)
        cost, logp, tour = self.model(coords, decode_type="sample")
        validate_tour(tour)
        self.assertEqual(cost.shape, (16,))
        self.assertEqual(logp.shape, (16,))
        self.assertTrue(torch.isfinite(cost).all())
        self.assertTrue(torch.isfinite(logp).all())

    def test_greedy_decoder_is_deterministic(self):
        coords = generate_tsp_batch(8, 8)
        c1, t1 = self.model.greedy(coords)
        c2, t2 = self.model.greedy(coords)
        torch.testing.assert_close(c1, c2)
        self.assertTrue(torch.equal(t1, t2))
        validate_tour(t1)

    def test_tour_length_hand_oracle(self):
        coords = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]
        )
        tour = torch.tensor([[0, 1, 2, 3]])
        self.assertTrue(math.isclose(float(tour_length(coords, tour)[0]), 4.0))

    def test_held_karp_matches_bruteforce(self):
        rng = np.random.default_rng(7)
        for n in (4, 5, 6, 7):
            coords = rng.random((n, 2))
            self.assertTrue(
                math.isclose(
                    held_karp_tsp(coords),
                    brute_force_tsp(coords),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
            )

    def test_nearest_neighbor_is_feasible(self):
        rng = np.random.default_rng(9)
        coords = rng.random((10, 2))
        cost, route = nearest_neighbor_tour(coords)
        self.assertEqual(sorted(route), list(range(10)))
        self.assertGreater(cost, 0.0)

    def test_reinforce_step_updates_parameters(self):
        coords = generate_tsp_batch(32, 8)
        baseline = GreedyRolloutBaseline(self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        before = [
            p.detach().clone()
            for p in self.model.parameters()
            if p.requires_grad
        ]
        metrics = reinforce_train_step(
            self.model,
            baseline,
            optimizer,
            coords,
        )
        after = [
            p.detach()
            for p in self.model.parameters()
            if p.requires_grad
        ]
        self.assertTrue(math.isfinite(metrics.loss))
        self.assertTrue(math.isfinite(metrics.grad_norm))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))

    def test_rollout_baseline_has_no_trainable_gradients(self):
        baseline = GreedyRolloutBaseline(self.model)
        self.assertTrue(
            all(not p.requires_grad for p in baseline.model.parameters())
        )
        coords = generate_tsp_batch(4, 7)
        values = baseline.evaluate(coords)
        self.assertEqual(values.shape, (4,))
        self.assertFalse(values.requires_grad)


if __name__ == "__main__":
    unittest.main()
