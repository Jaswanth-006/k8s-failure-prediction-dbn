import unittest
import numpy as np
import networkx as nx

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dbn_learner import DBNParameterLearner

class TestDBNParameterLearner(unittest.TestCase):

    def setUp(self):
        self.learner = DBNParameterLearner()

    def test_calibrate_emissions(self):
        # 10 Normal, 5 Degrading, 2 Critical (should fallback for critical)
        states = np.array([0]*10 + [1]*5 + [2]*2)
        
        # Anomaly scores:
        # Normal: mean 0.1, var 0.04
        normal_scores = np.random.normal(0.1, 0.2, 10)
        # Degrading: mean 2.0, var 0.25
        degrading_scores = np.random.normal(2.0, 0.5, 5)
        # Critical: just 2 samples
        critical_scores = np.array([5.0, 6.0])
        
        scores = np.concatenate([normal_scores, degrading_scores, critical_scores])
        
        mu, sigma = self.learner.calibrate_emissions(states, scores)
        
        self.assertAlmostEqual(mu[0], np.mean(normal_scores), places=3)
        expected_sigma_0 = np.sqrt(max(np.var(normal_scores, ddof=1), self.learner.min_variance))
        self.assertAlmostEqual(sigma[0], expected_sigma_0, places=3)
        
        self.assertAlmostEqual(mu[1], np.mean(degrading_scores), places=3)
        expected_sigma_1 = np.sqrt(max(np.var(degrading_scores, ddof=1), self.learner.min_variance))
        self.assertAlmostEqual(sigma[1], expected_sigma_1, places=3)
        
        # Critical should fallback to baseline since < 5 samples
        self.assertEqual(mu[2], self.learner.baseline_mu[2])
        self.assertEqual(sigma[2], self.learner.baseline_sigma[2])
        
    def test_calibrate_emissions_zero_variance(self):
        # All same value -> zero variance
        states = np.array([1]*10)
        scores = np.array([3.0]*10)
        
        mu, sigma = self.learner.calibrate_emissions(states, scores)
        self.assertEqual(mu[1], 3.0)
        # Should be bounded by min_variance
        self.assertAlmostEqual(sigma[1], np.sqrt(self.learner.min_variance), places=5)
        
    def test_calibrate_transitions(self):
        sequences = {
            "svcA": [0, 0, 1, 1, 2, 2, 2],
            "svcB": [0, 1, 2]
        }
        T = self.learner.calibrate_transitions(sequences)
        
        self.assertEqual(T.shape, (3, 3))
        # Check rows sum to 1
        np.testing.assert_allclose(np.sum(T, axis=1), [1.0, 1.0, 1.0])
        
    def test_calibrate_topological_influences(self):
        graph = nx.DiGraph()
        graph.add_edge("svcA", "svcB")
        
        # svcA is degrading, svcB goes from normal to degrading
        sequences = {
            "svcA": [0, 1, 1, 1, 1],
            "svcB": [0, 0, 1, 1, 1]
        }
        modifiers = self.learner.calibrate_topological_influences(sequences, graph)
        
        self.assertIn(1, modifiers)
        self.assertIn(2, modifiers)
        
        self.assertEqual(modifiers[1].shape, (3,))
        self.assertAlmostEqual(np.sum(modifiers[1]), 0.0, places=4)
        
if __name__ == "__main__":
    unittest.main()
