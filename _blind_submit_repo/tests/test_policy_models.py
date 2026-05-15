from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
GPYTORCH_AVAILABLE = importlib.util.find_spec("gpytorch") is not None


@unittest.skipUnless(TORCH_AVAILABLE and GPYTORCH_AVAILABLE, "requires torch and gpytorch")
class PolicyAndModelTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        from models import (
            DynamicObservationModel,
            MissingMechanism,
            MissingMechanismConfig,
            ObservationModelConfig,
        )
        from policy import InformationPolicyConfig, LazyGreedyCache, LazyGreedyOptimizer

        self.torch = torch
        self.DynamicObservationModel = DynamicObservationModel
        self.MissingMechanism = MissingMechanism
        self.MissingMechanismConfig = MissingMechanismConfig
        self.ObservationModelConfig = ObservationModelConfig
        self.InformationPolicyConfig = InformationPolicyConfig
        self.LazyGreedyCache = LazyGreedyCache
        self.LazyGreedyOptimizer = LazyGreedyOptimizer
        torch.manual_seed(0)

    def test_observation_link_learns_sensor_specific_offsets(self) -> None:
        config = self.ObservationModelConfig(num_sensor_types=2, link_fit_steps=60, link_learning_rate=0.05)
        model = self.DynamicObservationModel(config)

        z_mean = self.torch.zeros(8)
        sensor_type = self.torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=self.torch.long)
        y_true = self.torch.tensor([1.0, 1.1, 0.9, 1.0, 3.0, 3.1, 2.9, 3.0], dtype=self.torch.float32)

        history = model.fit_observation_link(
            y_true,
            z_mean,
            sensor_metadata={"sensor_type": sensor_type},
        )
        self.assertGreater(len(history["loss"]), 0)

        expected = model.expected_observation(
            z_mean,
            sensor_metadata={"sensor_type": sensor_type},
        )
        group0_mean = expected[sensor_type == 0].mean().item()
        group1_mean = expected[sensor_type == 1].mean().item()
        self.assertGreater(group1_mean - group0_mean, 1.0)

    def test_pi_ssd_embedding_and_self_supervised_loss(self) -> None:
        config = self.ObservationModelConfig(
            use_pi_ssd=True,
            diagnosis_mode="temporal_nwp",
            context_dim=1,
            nwp_context_index=0,
            diagnosis_embedding_dim=6,
            use_latent_ode=True,
            corruption_probability_start=0.05,
            corruption_probability_end=0.25,
        )
        model = self.DynamicObservationModel(config)
        z_mean = self.torch.zeros(6)
        z_var = self.torch.ones(6)
        y_true = self.torch.tensor([0.0, 0.2, 0.4, 1.0, 1.2, 1.4], dtype=self.torch.float32)
        context = self.torch.linspace(0.0, 1.0, 6).unsqueeze(-1)

        embedding = model.diagnosis_embedding(y_true, z_mean, z_var, context=context)
        early_components = model.diagnosis_self_supervision_components(
            y_true,
            z_mean,
            z_var,
            context=context,
            training_progress=0.0,
        )
        late_components = model.diagnosis_self_supervision_components(
            y_true,
            z_mean,
            z_var,
            context=context,
            training_progress=1.0,
        )
        loss = model.self_supervised_diagnosis_loss(
            y_true,
            z_mean,
            z_var,
            context=context,
            training_progress=1.0,
        )
        self.assertEqual(tuple(embedding.shape), (6, 6))
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertGreaterEqual(early_components["latent_ode_loss"].item(), 0.0)
        self.assertGreaterEqual(late_components["latent_ode_loss"].item(), 0.0)
        self.assertLess(
            early_components["mask_probability"].item(),
            late_components["mask_probability"].item(),
        )

    def test_fault_head_trains_and_scores_corrupted_signal_higher(self) -> None:
        config = self.ObservationModelConfig(
            use_pi_ssd=True,
            use_dbn=True,
            use_fault_head=True,
            link_fit_steps=40,
            fault_corruption_probability=0.3,
            fault_self_supervised_weight=0.4,
        )
        model = self.DynamicObservationModel(config)
        z_mean = self.torch.zeros(10)
        z_var = self.torch.ones(10)
        y_true = self.torch.linspace(0.0, 1.0, 10)

        history = model.fit_observation_link(y_true, z_mean, z_var=z_var)
        clean_probability = model.fault_probability(y_true, z_mean, z_var)
        corrupted_probability = model.fault_probability(y_true + 4.0, z_mean, z_var)
        self.assertIn("fault_self_supervised_loss", history)
        self.assertGreater(len(history["fault_self_supervised_loss"]), 0)
        self.assertEqual(tuple(clean_probability.shape), (10,))
        self.assertGreater(corrupted_probability.mean().item(), clean_probability.mean().item())

    def test_missingness_probability_responds_to_dynamic_features(self) -> None:
        config = self.MissingMechanismConfig(
            context_dim=1,
            num_sensor_types=2,
            num_sensor_groups=2,
            include_s=True,
            include_dynamic_residual=True,
            include_dynamic_threshold=True,
            include_normalized_residual=True,
            hidden_dim=16,
            trunk_depth=2,
        )
        model = self.MissingMechanism(config)

        z_mean = self.torch.zeros(6)
        z_var = self.torch.ones(6)
        context = self.torch.zeros(6, 1)
        metadata = {
            "sensor_type": self.torch.tensor([0, 0, 0, 1, 1, 1], dtype=self.torch.long),
            "sensor_group": self.torch.tensor([0, 0, 1, 1, 0, 1], dtype=self.torch.long),
        }
        target = self.torch.tensor([0, 0, 0, 1, 1, 1], dtype=self.torch.float32)

        optimizer = self.torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(
                target_missing=target,
                z_mean=z_mean,
                z_var=z_var,
                context=context,
                sensor_metadata=metadata,
                S=self.torch.tensor([0, 0, 0, 1, 1, 1], dtype=self.torch.float32),
                dynamic_residual=self.torch.tensor([0.1, 0.2, 0.1, 3.5, 4.0, 3.2], dtype=self.torch.float32),
                dynamic_threshold=self.torch.ones(6),
            )
            loss.backward()
            optimizer.step()

        low_risk = model.predict_proba(
            z_mean=z_mean[:3],
            z_var=z_var[:3],
            context=context[:3],
            sensor_metadata={
                "sensor_type": metadata["sensor_type"][:3],
                "sensor_group": metadata["sensor_group"][:3],
            },
            S=self.torch.zeros(3),
            dynamic_residual=self.torch.tensor([0.1, 0.2, 0.1], dtype=self.torch.float32),
            dynamic_threshold=self.torch.ones(3),
        )
        high_risk = model.predict_proba(
            z_mean=z_mean[3:],
            z_var=z_var[3:],
            context=context[3:],
            sensor_metadata={
                "sensor_type": metadata["sensor_type"][3:],
                "sensor_group": metadata["sensor_group"][3:],
            },
            S=self.torch.ones(3),
            dynamic_residual=self.torch.tensor([3.5, 4.0, 3.2], dtype=self.torch.float32),
            dynamic_threshold=self.torch.ones(3),
        )
        self.assertGreater(high_risk.mean().item(), low_risk.mean().item())

    def test_joint_variational_missingness_returns_loss_components(self) -> None:
        config = self.MissingMechanismConfig(
            context_dim=1,
            num_sensor_types=2,
            num_sensor_groups=2,
            include_s=True,
            inference_strategy="joint_variational",
            hidden_dim=16,
            encoder_hidden_dim=16,
            trunk_depth=2,
            encoder_depth=2,
            reconstruction_weight=0.5,
            kl_weight=0.1,
        )
        model = self.MissingMechanism(config)

        z_mean = self.torch.zeros(6)
        z_var = self.torch.ones(6)
        y = self.torch.tensor([0.0, 0.1, float("nan"), 1.2, 1.3, float("nan")], dtype=self.torch.float32)
        target = self.torch.tensor([0, 0, 1, 0, 0, 1], dtype=self.torch.float32)
        context = self.torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
        metadata = {
            "sensor_type": self.torch.tensor([0, 0, 0, 1, 1, 1], dtype=self.torch.long),
            "sensor_group": self.torch.tensor([0, 0, 1, 1, 0, 1], dtype=self.torch.long),
        }

        components = model.compute_loss_components(
            target_missing=target,
            z_mean=z_mean,
            z_var=z_var,
            y=y,
            context=context,
            sensor_metadata=metadata,
            S=self.torch.tensor([0, 0, 1, 0, 0, 1], dtype=self.torch.float32),
        )
        self.assertIn("total_loss", components)
        self.assertIn("reconstruction_loss", components)
        self.assertIn("kl_loss", components)
        self.assertIn("health_kl_loss", components)
        self.assertIn("health_reconstruction_loss", components)
        self.assertGreaterEqual(components["kl_loss"].item(), 0.0)
        self.assertGreaterEqual(components["reconstruction_loss"].item(), 0.0)
        self.assertGreaterEqual(components["health_kl_loss"].item(), 0.0)
        self.assertGreaterEqual(components["health_reconstruction_loss"].item(), 0.0)

        health = model.infer_health_posterior(
            z_mean=z_mean,
            z_var=z_var,
            y=y,
            context=context,
            sensor_metadata=metadata,
            S=self.torch.tensor([0, 0, 1, 0, 0, 1], dtype=self.torch.float32),
            target_missing=target,
        )
        self.assertEqual(tuple(health["health_mean"].shape), (6, config.health_latent_dim))
        self.assertEqual(tuple(health["health_var"].shape), (6, config.health_latent_dim))

    def test_joint_generative_missingness_returns_sampled_losses(self) -> None:
        config = self.MissingMechanismConfig(
            context_dim=1,
            num_sensor_types=2,
            num_sensor_groups=2,
            include_s=True,
            inference_strategy="joint_generative",
            hidden_dim=16,
            encoder_hidden_dim=16,
            trunk_depth=2,
            encoder_depth=2,
            reconstruction_weight=0.5,
            kl_weight=0.1,
            generative_samples=3,
            use_temporal_transition_prior=True,
            transition_time_index=0,
            transition_group_key="global",
        )
        model = self.MissingMechanism(config)

        X = self.torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
        z_mean = self.torch.zeros(6)
        z_var = self.torch.ones(6)
        y = self.torch.tensor([0.0, 0.1, float("nan"), 1.2, 1.3, float("nan")], dtype=self.torch.float32)
        target = self.torch.tensor([0, 0, 1, 0, 0, 1], dtype=self.torch.float32)
        context = self.torch.linspace(0.0, 1.0, 6).unsqueeze(-1)
        metadata = {
            "sensor_type": self.torch.tensor([0, 0, 0, 1, 1, 1], dtype=self.torch.long),
            "sensor_group": self.torch.tensor([0, 0, 1, 1, 0, 1], dtype=self.torch.long),
        }

        components = model.compute_loss_components(
            target_missing=target,
            z_mean=z_mean,
            z_var=z_var,
            y=y,
            context=context,
            sensor_metadata=metadata,
            S=self.torch.tensor([0, 0, 1, 0, 0, 1], dtype=self.torch.float32),
            X=X,
        )
        proba = model.predict_proba(
            z_mean=z_mean,
            z_var=z_var,
            y=y,
            context=context,
            sensor_metadata=metadata,
            S=self.torch.tensor([0, 0, 1, 0, 0, 1], dtype=self.torch.float32),
            X=X,
        )
        self.assertIn("total_loss", components)
        self.assertGreaterEqual(components["reconstruction_loss"].item(), 0.0)
        self.assertGreaterEqual(components["kl_loss"].item(), 0.0)
        self.assertGreaterEqual(components["transition_loss"].item(), 0.0)
        self.assertTrue(self.torch.all(proba > 0.0))
        self.assertTrue(self.torch.all(proba < 1.0))

    def test_mi_proxy_prefers_non_redundant_candidate(self) -> None:
        config = self.InformationPolicyConfig(
            utility_surrogate="mi_proxy",
            selection_mode="budget",
            planning_strategy="lazy_greedy",
            lengthscale=0.5,
            observation_noise=0.1,
        )
        optimizer = self.LazyGreedyOptimizer(config)
        candidate_x = self.torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]],
            dtype=self.torch.float32,
        )
        candidate_var = self.torch.ones(3)
        candidate_cost = self.torch.ones(3)
        cache = self.LazyGreedyCache(
            max_similarity=self.torch.zeros(3),
            log_conditional_variance_factor=self.torch.zeros(3),
            selected_mask=self.torch.zeros(3, dtype=self.torch.bool),
            current_cost=0.0,
        )
        optimizer.update_cache(cache, candidate_x, candidate_var, selected_index=0)

        duplicate_gain = optimizer.marginal_gain(1, candidate_var, candidate_cost, cache).item()
        distinct_gain = optimizer.marginal_gain(2, candidate_var, candidate_cost, cache).item()
        self.assertLess(duplicate_gain, distinct_gain)

    def test_non_myopic_rollout_prefers_high_volatility_context(self) -> None:
        config = self.InformationPolicyConfig(
            utility_surrogate="mi_proxy",
            planning_strategy="non_myopic_rollout",
            planning_horizon=4,
            future_discount=0.9,
            lookahead_strength=0.8,
            future_context_index=0,
            future_volatility_weight=0.5,
        )
        optimizer = self.LazyGreedyOptimizer(config)
        candidate_var = self.torch.ones(2)
        candidate_cost = self.torch.ones(2)
        candidate_context = self.torch.tensor([[0.1], [8.0]], dtype=self.torch.float32)
        scores = optimizer.score_candidate(
            candidate_var,
            candidate_cost,
            conditional_variance=candidate_var,
            context_features=candidate_context,
        )
        self.assertGreater(scores[1].item(), scores[0].item())

    def test_ppo_warmstart_trains_policy_prior_and_records_stats(self) -> None:
        config = self.InformationPolicyConfig(
            utility_surrogate="mi_proxy",
            planning_strategy="ppo_warmstart",
            ppo_epochs=3,
            ppo_max_candidates=8,
            ppo_policy_weight=0.5,
        )
        optimizer = self.LazyGreedyOptimizer(config)
        candidate_x = self.torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype=self.torch.float32,
        )
        candidate_var = self.torch.tensor([0.2, 0.4, 1.5, 1.8], dtype=self.torch.float32)
        candidate_cost = self.torch.ones(4)
        stats = optimizer.warm_start_policy(
            candidate_x,
            candidate_cost,
            candidate_var,
            max_selections=2,
        )
        prior = optimizer._policy_prior(candidate_x, candidate_cost, candidate_var)
        selection = optimizer.select(candidate_x, candidate_cost, candidate_var, max_selections=2)
        self.assertIn("policy_training_loss", stats)
        self.assertEqual(prior.shape[0], 4)
        self.assertIn("policy_stats", selection)
        self.assertEqual(selection["planning_strategy"], "ppo_warmstart")

    def test_ppo_online_records_rollout_training_stats(self) -> None:
        config = self.InformationPolicyConfig(
            utility_surrogate="mi_proxy",
            planning_strategy="ppo_online",
            ppo_epochs=2,
            ppo_rollout_episodes=2,
            ppo_max_candidates=8,
            ppo_policy_weight=0.5,
        )
        optimizer = self.LazyGreedyOptimizer(config)
        candidate_x = self.torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype=self.torch.float32,
        )
        candidate_var = self.torch.tensor([0.2, 0.4, 1.5, 1.8], dtype=self.torch.float32)
        candidate_cost = self.torch.ones(4)
        stats = optimizer.online_train_policy(
            candidate_x,
            candidate_cost,
            candidate_var,
            max_selections=2,
        )
        selection = optimizer.select(candidate_x, candidate_cost, candidate_var, max_selections=2)
        self.assertIn("policy_rollout_episodes", stats)
        self.assertIn("policy_rollout_mean_reward", stats)
        self.assertIn("policy_stats", selection)
        self.assertEqual(selection["planning_strategy"], "ppo_online")

    def test_similarity_ignores_temporal_dimensions_for_placement(self) -> None:
        optimizer = self.LazyGreedyOptimizer(self.InformationPolicyConfig(lengthscale=0.5))
        candidate_x = self.torch.tensor(
            [
                [35.0, 128.0, 0.0, 0.0],
                [35.0, 128.0, 24.0, 1.0],
            ],
            dtype=self.torch.float32,
        )
        similarity = optimizer._similarity_to_anchor(candidate_x, candidate_x[0])
        self.assertAlmostEqual(similarity[1].item(), 1.0, places=6)

    def test_route_penalty_prefers_closer_follow_up_selection(self) -> None:
        config = self.InformationPolicyConfig(
            utility_surrogate="mi_proxy",
            planning_strategy="lazy_greedy",
            route_distance_weight=0.5,
            lengthscale=1.0,
        )
        optimizer = self.LazyGreedyOptimizer(config)
        candidate_x = self.torch.tensor(
            [[0.0, 0.0], [0.2, 0.2], [5.0, 5.0]],
            dtype=self.torch.float32,
        )
        candidate_var = self.torch.tensor([1.0, 1.0, 1.0], dtype=self.torch.float32)
        candidate_cost = self.torch.ones(3)
        cache = self.LazyGreedyCache(
            max_similarity=self.torch.zeros(3),
            log_conditional_variance_factor=self.torch.zeros(3),
            selected_mask=self.torch.zeros(3, dtype=self.torch.bool),
            current_cost=0.0,
        )
        optimizer.update_cache(cache, candidate_x, candidate_var, selected_index=0)
        near_gain = optimizer.marginal_gain(1, candidate_var, candidate_cost, cache, candidate_x=candidate_x).item()
        far_gain = optimizer.marginal_gain(2, candidate_var, candidate_cost, cache, candidate_x=candidate_x).item()
        self.assertGreater(near_gain, far_gain)

    def test_ratio_mode_uses_route_aware_operational_cost(self) -> None:
        config = self.InformationPolicyConfig(
            utility_surrogate="variance",
            planning_strategy="lazy_greedy",
            selection_mode="ratio",
            route_distance_weight=1.0,
            lengthscale=1.0,
        )
        optimizer = self.LazyGreedyOptimizer(config)
        near_gain = optimizer.score_candidate(
            self.torch.tensor([1.0]),
            self.torch.tensor([1.0]),
            conditional_variance=self.torch.tensor([1.0]),
            route_distance=self.torch.tensor([1.0]),
        ).item()
        far_gain = optimizer.score_candidate(
            self.torch.tensor([1.0]),
            self.torch.tensor([1.0]),
            conditional_variance=self.torch.tensor([1.0]),
            route_distance=self.torch.tensor([5.0]),
        ).item()
        self.assertGreater(near_gain, far_gain)

    def test_select_reorders_route_to_reduce_routing_cost(self) -> None:
        config = self.InformationPolicyConfig(
            utility_surrogate="variance",
            planning_strategy="lazy_greedy",
            selection_mode="budget",
            route_distance_weight=0.0,
            budget=10.0,
        )
        optimizer = self.LazyGreedyOptimizer(config)
        candidate_x = self.torch.tensor(
            [[0.0, 0.0], [10.0, 0.0], [1.0, 0.0]],
            dtype=self.torch.float32,
        )
        candidate_var = self.torch.tensor([3.0, 2.0, 1.0], dtype=self.torch.float32)
        candidate_cost = self.torch.ones(3)
        selection = optimizer.select(candidate_x, candidate_cost, candidate_var, max_selections=3)
        self.assertEqual(selection["selected_indices"], [0, 2, 1])
        self.assertLess(selection["routing_cost"], 20.0)

    def test_policy_prior_only_screens_large_candidate_pools(self) -> None:
        config = self.InformationPolicyConfig(
            utility_surrogate="variance",
            planning_strategy="ppo_warmstart",
            selection_mode="budget",
            ppo_max_candidates=4,
            ppo_policy_weight=0.25,
        )
        optimizer = self.LazyGreedyOptimizer(config)
        candidate_x = self.torch.arange(20, dtype=self.torch.float32).view(10, 2)
        candidate_var = self.torch.linspace(1.0, 2.0, steps=10)
        candidate_cost = self.torch.ones(10)
        cache = optimizer._initialize_cache(candidate_x)
        upper_bounds = optimizer._current_upper_bounds(
            candidate_x,
            candidate_cost,
            candidate_var,
            cache=cache,
        )
        policy_prior = self.torch.linspace(0.1, 1.0, steps=10)
        screened = optimizer._screen_with_policy_prior(
            upper_bounds,
            policy_prior,
            candidate_x,
            candidate_cost,
            cache=cache,
            active_limit=2,
        )
        finite_mask = self.torch.isfinite(screened)
        self.assertEqual(int(finite_mask.sum().item()), 4)
        self.assertTrue(
            self.torch.allclose(
                screened[finite_mask],
                upper_bounds[finite_mask],
                atol=1e-6,
                rtol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
