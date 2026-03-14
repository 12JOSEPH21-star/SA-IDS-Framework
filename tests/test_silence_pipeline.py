from __future__ import annotations

import importlib.util
import math
import unittest
from unittest import mock


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
GPYTORCH_AVAILABLE = importlib.util.find_spec("gpytorch") is not None


@unittest.skipUnless(TORCH_AVAILABLE and GPYTORCH_AVAILABLE, "requires torch and gpytorch")
class SilenceAwarePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        from models import MissingMechanismConfig, ObservationModelConfig, SparseGPConfig
        from pipeline import (
            MissingnessTrainingConfig,
            SilenceAwareIDS,
            SilenceAwareIDSConfig,
            StateTrainingConfig,
        )
        from reliability import ConformalConfig

        self.torch = torch
        self.SilenceAwareIDS = SilenceAwareIDS
        self.SilenceAwareIDSConfig = SilenceAwareIDSConfig
        self.SparseGPConfig = SparseGPConfig
        self.MissingMechanismConfig = MissingMechanismConfig
        self.ObservationModelConfig = ObservationModelConfig
        self.StateTrainingConfig = StateTrainingConfig
        self.MissingnessTrainingConfig = MissingnessTrainingConfig
        self.ConformalConfig = ConformalConfig

        torch.manual_seed(0)
        self.x_train = torch.linspace(0.0, 1.0, 12).unsqueeze(-1)
        self.y_train = torch.sin(2.0 * math.pi * self.x_train).squeeze(-1)
        self.x_eval = torch.linspace(0.05, 0.95, 8).unsqueeze(-1)
        self.y_eval = torch.sin(2.0 * math.pi * self.x_eval).squeeze(-1)
        self.missing_indicator = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=torch.float32)
        self.sensor_metadata = {
            "sensor_type": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long),
            "sensor_group": torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long),
        }

    def _build_pipeline(
        self,
        *,
        use_m2: bool = True,
        use_m3: bool = True,
        use_m5: bool = False,
        observation: object | None = None,
        missingness: object | None = None,
        reliability: object | None = None,
        homogeneous_missingness: bool = False,
        sensor_conditional_missingness: bool = True,
    ) -> object:
        config = self.SilenceAwareIDSConfig(
            state=self.SparseGPConfig(input_dim=1, inducing_points=4),
            observation=(
                self.ObservationModelConfig() if observation is None else observation
            ),
            missingness=(
                self.MissingMechanismConfig(
                    num_sensor_types=2,
                    num_sensor_groups=2,
                    include_s=True,
                )
                if missingness is None
                else missingness
            ),
            reliability=(
                self.ConformalConfig() if reliability is None else reliability
            ),
            state_training=self.StateTrainingConfig(epochs=1, batch_size=4),
            missingness_training=self.MissingnessTrainingConfig(epochs=1, batch_size=4),
            use_m2=use_m2,
            use_m3=use_m3,
            use_m5=use_m5,
            homogeneous_missingness=homogeneous_missingness,
            sensor_conditional_missingness=sensor_conditional_missingness,
        )
        return self.SilenceAwareIDS(config)

    def test_build_ablation_configs_contains_core_variants(self) -> None:
        pipeline = self._build_pipeline()
        variants = pipeline.build_ablation_configs()
        self.assertIn("base_gp_only", variants)
        self.assertIn("gp_plus_sensor_conditional_missingness", variants)
        self.assertIn("gp_plus_joint_variational_missingness", variants)
        self.assertIn("gp_plus_joint_jvi_training", variants)
        self.assertIn("gp_plus_pattern_mixture_missingness", variants)
        self.assertIn("relational_reliability_baseline", variants)
        self.assertIn("myopic_policy_baseline", variants)
        self.assertIn("ppo_warmstart_baseline", variants)
        self.assertIn("rollout_policy_baseline", variants)
        self.assertIn("variance_policy_baseline", variants)
        self.assertFalse(variants["gp_plus_pattern_mixture_missingness"].use_m5)
        self.assertEqual(
            variants["gp_plus_joint_variational_missingness"].missingness.inference_strategy,
            "joint_variational",
        )
        self.assertEqual(variants["gp_plus_joint_variational_missingness"].state_training.training_strategy, "sequential")
        self.assertEqual(variants["gp_plus_joint_jvi_training"].state_training.training_strategy, "joint_variational")
        self.assertEqual(variants["myopic_policy_baseline"].policy.planning_strategy, "lazy_greedy")
        self.assertEqual(variants["ppo_warmstart_baseline"].policy.planning_strategy, "ppo_warmstart")
        self.assertEqual(variants["rollout_policy_baseline"].policy.planning_strategy, "non_myopic_rollout")
        self.assertEqual(variants["full_model"].policy.planning_strategy, "ppo_online")

    def test_sparse_gp_requires_explicit_dims_for_high_dim_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.SparseGPConfig(input_dim=7, inducing_points=4)

    def test_fit_and_sensitivity_interfaces(self) -> None:
        pipeline = self._build_pipeline()
        summary = pipeline.fit(
            self.x_train,
            self.y_train,
            missing_indicator_train=self.missing_indicator,
            sensor_metadata_train=self.sensor_metadata,
        )
        self.assertIn("loss", summary.state_history)
        self.assertIsNotNone(summary.missingness_history)
        self.assertIn("latent_ode_loss", summary.observation_history)
        self.assertIn("curriculum_mask_probability", summary.observation_history)

        sensitivity = pipeline.run_missingness_sensitivity_analysis(
            self.x_eval,
            self.y_eval,
            logit_scales=[0.5, 1.0],
            sensor_metadata={
                "sensor_type": self.sensor_metadata["sensor_type"][: self.x_eval.shape[0]],
                "sensor_group": self.sensor_metadata["sensor_group"][: self.x_eval.shape[0]],
            },
        )
        self.assertIn("logit_scale_0.500", sensitivity)
        self.assertIn("rmse", sensitivity["logit_scale_1.000"])

    def test_run_ablation_suite_subset(self) -> None:
        pipeline = self._build_pipeline()
        outcomes = pipeline.run_ablation_suite(
            self.x_train,
            self.y_train,
            self.x_eval,
            self.y_eval,
            missing_indicator_train=self.missing_indicator,
            sensor_metadata_train=self.sensor_metadata,
            sensor_metadata_eval={
                "sensor_type": self.sensor_metadata["sensor_type"][: self.x_eval.shape[0]],
                "sensor_group": self.sensor_metadata["sensor_group"][: self.x_eval.shape[0]],
            },
            variant_names=["base_gp_only", "gp_plus_sensor_conditional_missingness"],
        )
        self.assertIn("base_gp_only", outcomes)
        self.assertIn("gp_plus_sensor_conditional_missingness", outcomes)
        self.assertIn("rmse", outcomes["base_gp_only"].metrics)

    def test_forward_handles_y_without_m2(self) -> None:
        pipeline = self.SilenceAwareIDS(
            self.SilenceAwareIDSConfig(
                state=self.SparseGPConfig(input_dim=1, inducing_points=4),
                missingness=self.MissingMechanismConfig(
                    num_sensor_types=2,
                    num_sensor_groups=2,
                    include_s=True,
                ),
                state_training=self.StateTrainingConfig(epochs=1, batch_size=4),
                missingness_training=self.MissingnessTrainingConfig(epochs=1, batch_size=4),
                use_m2=False,
                use_m3=True,
                use_m5=False,
                homogeneous_missingness=False,
                sensor_conditional_missingness=True,
            )
        )
        pipeline.fit_state_model(self.x_train, self.y_train)
        outputs = pipeline(
            self.x_eval,
            y=self.y_eval,
            sensor_metadata={
                "sensor_type": self.sensor_metadata["sensor_type"][: self.x_eval.shape[0]],
                "sensor_group": self.sensor_metadata["sensor_group"][: self.x_eval.shape[0]],
            },
        )
        self.assertIsNone(outputs["dynamic_silence"])
        self.assertIsNotNone(outputs["missingness_proba"])

    def test_predict_state_defaults_to_latent_summary(self) -> None:
        pipeline = self._build_pipeline()
        pipeline.fit_state_model(self.x_train, self.y_train)
        pipeline.state_likelihood.noise = 0.5
        _, latent_var = pipeline.predict_state(self.x_eval, batch_size=4)
        _, observation_var = pipeline.predict_state(
            self.x_eval,
            batch_size=4,
            include_observation_noise=True,
        )
        self.assertTrue(self.torch.all(observation_var >= latent_var))
        self.assertGreater((observation_var - latent_var).mean().item(), 0.0)

    def test_detect_silence_propagates_batch_size(self) -> None:
        pipeline = self._build_pipeline()
        pipeline.fit_state_model(self.x_train, self.y_train)
        original_predict_state = pipeline.predict_state

        def wrapped_predict_state(X, *, batch_size=None, include_observation_noise=False):
            self.assertEqual(batch_size, 3)
            return original_predict_state(
                X,
                batch_size=batch_size,
                include_observation_noise=include_observation_noise,
            )

        with mock.patch.object(pipeline, "predict_state", side_effect=wrapped_predict_state):
            pipeline.detect_silence(self.x_eval, self.y_eval, batch_size=3)

    def test_detect_silence_uses_observation_variance(self) -> None:
        pipeline = self._build_pipeline()

        def fake_predict_state(X, *, batch_size=None, include_observation_noise=False):
            self.assertTrue(include_observation_noise)
            return self.torch.zeros(X.shape[0]), self.torch.ones(X.shape[0])

        with mock.patch.object(pipeline, "predict_state", side_effect=fake_predict_state):
            outputs = pipeline.detect_silence(self.x_eval, self.y_eval, batch_size=2)
        self.assertIn("available", outputs)
        self.assertIn("diagnostic_score", outputs)
        self.assertIn("diagnosis_embedding", outputs)
        self.assertIn("sensor_state_probs", outputs)
        self.assertIn("sensor_state_labels", outputs)
        self.assertEqual(outputs["diagnosis_embedding"].shape[0], self.x_eval.shape[0])
        self.assertEqual(outputs["sensor_state_probs"].shape[0], self.x_eval.shape[0])
        self.assertTrue(outputs["available"].all().item())

    def test_temporal_nwp_diagnosis_exposes_sequence_score(self) -> None:
        observation = self.ObservationModelConfig(
            diagnosis_mode="temporal_nwp",
            context_dim=1,
            nwp_context_index=0,
            nwp_anchor_weight=0.5,
        )
        pipeline = self._build_pipeline(use_m3=False, observation=observation)

        def fake_predict_state(X, *, batch_size=None, include_observation_noise=False):
            return self.torch.zeros(X.shape[0]), self.torch.ones(X.shape[0])

        context = self.torch.linspace(0.0, 1.0, self.x_eval.shape[0]).unsqueeze(-1)
        y_eval = self.torch.tensor([0.0, 0.1, 0.2, 2.0, 2.1, 2.2, 2.3, 2.4])
        with mock.patch.object(pipeline, "predict_state", side_effect=fake_predict_state):
            outputs = pipeline.detect_silence(self.x_eval, y_eval, context=context, batch_size=2)
        self.assertGreater(outputs["diagnostic_score"][3].item(), outputs["diagnostic_score"][1].item())

    def test_predict_missingness_propagates_dynamic_availability(self) -> None:
        pipeline = self._build_pipeline()
        fake_silence = {
            "dynamic_silence": self.torch.tensor([1, 0, 1, 0, 0, 0, 0, 0], dtype=self.torch.bool),
            "residuals": self.torch.arange(8, dtype=self.torch.float32),
            "threshold": self.torch.ones(8),
            "available": self.torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=self.torch.bool),
        }

        def fake_predict_state(X, *, batch_size=None, include_observation_noise=False):
            return self.torch.zeros(X.shape[0]), self.torch.ones(X.shape[0])

        def fake_predict_proba(**kwargs):
            self.assertTrue(self.torch.equal(kwargs["dynamic_feature_available"], fake_silence["available"].float()))
            return self.torch.full((8,), 0.25)

        with mock.patch.object(pipeline, "detect_silence", return_value=fake_silence):
            with mock.patch.object(pipeline, "predict_state", side_effect=fake_predict_state):
                with mock.patch.object(pipeline.missingness_model, "predict_proba", side_effect=fake_predict_proba):
                    probabilities = pipeline.predict_missingness(self.x_eval, y=self.y_eval, batch_size=2)
        self.assertEqual(probabilities.shape[0], 8)

    def test_select_sensors_uses_deployment_safe_missingness_features(self) -> None:
        missingness = self.MissingMechanismConfig(
            num_sensor_types=2,
            num_sensor_groups=2,
            include_s=True,
            use_dynamic_features_for_policy=False,
        )
        pipeline = self._build_pipeline(missingness=missingness)

        def fake_predict_state(X, *, batch_size=None, include_observation_noise=False):
            return self.torch.zeros(X.shape[0]), self.torch.ones(X.shape[0])

        def fake_predict_proba(**kwargs):
            self.assertIsNone(kwargs["S"])
            self.assertTrue(torch.equal(kwargs["dynamic_feature_available"], self.torch.zeros(8)))
            return self.torch.zeros(8)

        torch = self.torch
        with mock.patch.object(pipeline, "predict_state", side_effect=fake_predict_state):
            with mock.patch.object(pipeline.missingness_model, "predict_proba", side_effect=fake_predict_proba):
                selection = pipeline.select_sensors(
                    self.x_eval,
                    self.torch.ones(8),
                    S=self.torch.ones(8),
                    batch_size=2,
                    max_selections=2,
                )
        self.assertEqual(len(selection["selected_indices"]), 2)

    def test_quantile_dynamic_silence_requires_calibration_split(self) -> None:
        pipeline = self._build_pipeline(
            use_m3=False,
            observation=self.ObservationModelConfig(threshold_mode="quantile"),
        )
        with self.assertRaises(ValueError):
            pipeline.fit(self.x_train, self.y_train)

    def test_missingness_config_preserves_dynamic_feature_flags(self) -> None:
        pipeline = self.SilenceAwareIDS(
            self.SilenceAwareIDSConfig(
                state=self.SparseGPConfig(input_dim=1, inducing_points=4),
                missingness=self.MissingMechanismConfig(
                    include_s=True,
                    include_dynamic_residual=False,
                    include_dynamic_threshold=False,
                    include_normalized_residual=False,
                    positive_class_weight=2.5,
                ),
                state_training=self.StateTrainingConfig(epochs=1, batch_size=4),
                missingness_training=self.MissingnessTrainingConfig(epochs=1, batch_size=4),
                use_m2=True,
                use_m3=True,
                use_m5=False,
                homogeneous_missingness=True,
                sensor_conditional_missingness=False,
            )
        )
        self.assertFalse(pipeline.missingness_model.config.include_dynamic_residual)
        self.assertFalse(pipeline.missingness_model.config.include_dynamic_threshold)
        self.assertFalse(pipeline.missingness_model.config.include_normalized_residual)
        self.assertEqual(pipeline.missingness_model.config.positive_class_weight, 2.5)

    def test_missingness_aware_state_summary_uses_joint_variational_adapter(self) -> None:
        missingness = self.MissingMechanismConfig(
            num_sensor_types=2,
            num_sensor_groups=2,
            include_s=True,
            inference_strategy="joint_variational",
        )
        pipeline = self._build_pipeline(missingness=missingness)

        def fake_predict_state(X, *, batch_size=None, include_observation_noise=False):
            return self.torch.zeros(X.shape[0]), self.torch.ones(X.shape[0])

        def fake_infer_latent_posterior(**kwargs):
            size = kwargs["z_mean"].shape[0]
            return (
                self.torch.full((size,), 2.0),
                self.torch.full((size,), 3.0),
                {"kl_loss": self.torch.tensor(0.0), "reconstruction_loss": self.torch.tensor(0.0)},
            )

        with mock.patch.object(pipeline, "predict_state", side_effect=fake_predict_state):
            with mock.patch.object(
                pipeline.missingness_model,
                "infer_latent_posterior",
                side_effect=fake_infer_latent_posterior,
            ):
                with mock.patch.object(
                    pipeline,
                    "predict_missingness",
                    return_value=self.torch.zeros(self.x_eval.shape[0]),
                ):
                    mean, var, _ = pipeline.missingness_aware_state_summary(
                        self.x_eval,
                        y=self.y_eval,
                        sensor_metadata={
                            "sensor_type": self.sensor_metadata["sensor_type"][: self.x_eval.shape[0]],
                            "sensor_group": self.sensor_metadata["sensor_group"][: self.x_eval.shape[0]],
                        },
                        batch_size=2,
                    )
        self.assertTrue(self.torch.allclose(mean, self.torch.full_like(mean, 2.0)))
        self.assertTrue(self.torch.allclose(var, self.torch.full_like(var, 3.0)))

    def test_calibrate_reliability_uses_observation_target(self) -> None:
        pipeline = self._build_pipeline(use_m5=True)

        def fake_predict_state(X, *, batch_size=None, include_observation_noise=False):
            self.assertTrue(include_observation_noise)
            return self.torch.zeros(X.shape[0]), self.torch.ones(X.shape[0])

        with mock.patch.object(pipeline, "predict_state", side_effect=fake_predict_state):
            q_hat = pipeline.calibrate_reliability(self.x_eval, self.y_eval, batch_size=2)
        self.assertTrue(self.torch.is_tensor(q_hat))

    def test_adaptive_reliability_reports_final_epsilon(self) -> None:
        reliability = self.ConformalConfig(mode="adaptive", adaptation_rate=0.1)
        pipeline = self._build_pipeline(use_m5=True, reliability=reliability)

        def fake_predict_state(X, *, batch_size=None, include_observation_noise=False):
            return self.torch.zeros(X.shape[0]), self.torch.ones(X.shape[0])

        with mock.patch.object(pipeline, "predict_state", side_effect=fake_predict_state):
            pipeline.calibrate_reliability(self.x_eval, self.y_eval, batch_size=2)
            metrics = pipeline.evaluate_predictions(self.x_eval, self.y_eval, batch_size=2)
        self.assertIn("final_adaptive_epsilon", metrics)

    def test_relational_adaptive_reliability_reports_neighbor_error(self) -> None:
        reliability = self.ConformalConfig(
            mode="relational_adaptive",
            adaptation_rate=0.1,
            relational_neighbor_weight=0.5,
        )
        pipeline = self._build_pipeline(use_m5=True, reliability=reliability)

        def fake_predict_state(X, *, batch_size=None, include_observation_noise=False):
            return self.torch.zeros(X.shape[0]), self.torch.ones(X.shape[0])

        with mock.patch.object(pipeline, "predict_state", side_effect=fake_predict_state):
            pipeline.calibrate_reliability(self.x_eval, self.y_eval, batch_size=2)
            metrics = pipeline.evaluate_predictions(self.x_eval, self.y_eval, batch_size=2)
        self.assertIn("final_adaptive_epsilon", metrics)
        self.assertIn("mean_neighbor_error", metrics)

    def test_graph_corel_reliability_reports_local_quantiles(self) -> None:
        reliability = self.ConformalConfig(
            mode="graph_corel",
            graph_training_steps=2,
            graph_k_neighbors=2,
            graph_message_passing_steps=2,
        )
        pipeline = self._build_pipeline(use_m5=True, reliability=reliability)

        def fake_predict_state(X, *, batch_size=None, include_observation_noise=False):
            return self.torch.zeros(X.shape[0]), self.torch.ones(X.shape[0])

        with mock.patch.object(pipeline, "predict_state", side_effect=fake_predict_state):
            pipeline.calibrate_reliability(self.x_eval, self.y_eval, batch_size=2)
            metrics = pipeline.evaluate_predictions(self.x_eval, self.y_eval, batch_size=2)
            lower, upper, metadata = pipeline.predict_interval(self.x_eval, y=self.y_eval, batch_size=2)
        self.assertIn("mean_graph_quantile", metrics)
        self.assertIn("q_hat", metadata)
        self.assertEqual(metadata["graph_message_passing_steps"], 2.0)
        self.assertEqual(lower.shape[0], self.x_eval.shape[0])
        self.assertEqual(upper.shape[0], self.x_eval.shape[0])

    def test_ablation_report_flags_are_variant_specific(self) -> None:
        pipeline = self._build_pipeline()
        pattern_variant = pipeline.spawn_ablation_variant("gp_plus_pattern_mixture_missingness")
        relational_variant = pipeline.spawn_ablation_variant("relational_reliability_baseline")
        myopic_variant = pipeline.spawn_ablation_variant("myopic_policy_baseline")
        ppo_warm_variant = pipeline.spawn_ablation_variant("ppo_warmstart_baseline")
        joint_jvi_variant = pipeline.spawn_ablation_variant("gp_plus_joint_jvi_training")
        rollout_variant = pipeline.spawn_ablation_variant("rollout_policy_baseline")
        full_variant = pipeline.spawn_ablation_variant("full_model")

        pattern_flags = pattern_variant.ablation_report()["comparison_grid"]
        relational_flags = relational_variant.ablation_report()["comparison_grid"]
        myopic_flags = myopic_variant.ablation_report()["comparison_grid"]
        ppo_warm_flags = ppo_warm_variant.ablation_report()["comparison_grid"]
        joint_jvi_flags = joint_jvi_variant.ablation_report()["comparison_grid"]
        rollout_flags = rollout_variant.ablation_report()["comparison_grid"]
        full_flags = full_variant.ablation_report()["comparison_grid"]
        full_report = full_variant.ablation_report()

        self.assertTrue(pattern_flags["gp_plus_pattern_mixture_missingness"])
        self.assertFalse(pattern_flags["gp_plus_conformal_reliability"])
        self.assertFalse(pattern_flags["full_model"])
        self.assertTrue(relational_flags["relational_reliability_baseline"])
        self.assertFalse(relational_flags["gp_plus_conformal_reliability"])
        self.assertTrue(myopic_flags["myopic_policy_baseline"])
        self.assertFalse(myopic_flags["full_model"])
        self.assertTrue(ppo_warm_flags["ppo_warmstart_baseline"])
        self.assertFalse(ppo_warm_flags["full_model"])
        self.assertTrue(joint_jvi_flags["gp_plus_joint_jvi_training"])
        self.assertFalse(joint_jvi_flags["gp_plus_joint_variational_missingness"])
        self.assertTrue(rollout_flags["rollout_policy_baseline"])
        self.assertFalse(rollout_flags["full_model"])
        self.assertTrue(full_flags["full_model"])
        self.assertFalse(full_flags["gp_plus_conformal_reliability"])
        self.assertFalse(full_flags["gp_plus_sensor_conditional_missingness"])
        self.assertFalse(full_flags["gp_plus_joint_variational_missingness"])
        self.assertIn("diagnosis_mode", full_report)
        self.assertIn("diagnosis_representation", full_report)
        self.assertIn("diagnosis_temporal_model", full_report)
        self.assertIn("diagnosis_curriculum", full_report)
        self.assertIn("diagnosis_latent_dynamics", full_report)
        self.assertIn("missingness_sensor_health_latent", full_report)
        self.assertIn("reliability_mode", full_report)
        self.assertIn("reliability_relational", full_report)
        self.assertIn("reliability_graph_corel", full_report)
        self.assertIn("reliability_graph_message_passing_steps", full_report)
        self.assertIn("missingness_inference_strategy", full_report)
        self.assertIn("policy_planning_strategy", full_report)

    def test_forward_exposes_latent_and_observation_spaces(self) -> None:
        pipeline = self._build_pipeline(use_m5=False)
        pipeline.fit_state_model(self.x_train, self.y_train)
        outputs = pipeline(self.x_eval, y=self.y_eval, batch_size=2)

        self.assertEqual(outputs["state_target"], "latent")
        self.assertEqual(outputs["observation_target"], "observation")
        self.assertIsNotNone(outputs["observation_mean"])
        self.assertIsNotNone(outputs["observation_var"])
        self.assertIsNotNone(outputs["diagnosis_embedding"])
        self.assertIsNotNone(outputs["sensor_state_probs"])
        self.assertIsNotNone(outputs["sensor_state_labels"])
        self.assertIsNotNone(outputs["health_latent_mean"])
        self.assertIsNotNone(outputs["health_latent_var"])
        self.assertEqual(outputs["health_latent_mean"].shape[0], self.x_eval.shape[0])
        self.assertTrue(self.torch.all(outputs["observation_var"] >= outputs["state_var"]))

    def test_fit_missingness_tracks_health_latent_losses(self) -> None:
        missingness = self.MissingMechanismConfig(
            num_sensor_types=2,
            num_sensor_groups=2,
            include_s=True,
            inference_strategy="joint_variational",
            use_sensor_health_latent=True,
        )
        pipeline = self._build_pipeline(missingness=missingness)
        pipeline.fit_state_model(self.x_train, self.y_train)
        history = pipeline.fit_missingness_model(
            self.x_train,
            self.missing_indicator,
            y=self.y_train,
            sensor_metadata=self.sensor_metadata,
            batch_size=4,
        )
        self.assertIn("health_kl_loss", history)
        self.assertIn("health_reconstruction_loss", history)

    def test_joint_jvi_training_populates_state_and_missingness_histories(self) -> None:
        missingness = self.MissingMechanismConfig(
            num_sensor_types=2,
            num_sensor_groups=2,
            include_s=True,
            inference_strategy="joint_variational",
            use_sensor_health_latent=True,
        )
        pipeline = self.SilenceAwareIDS(
            self.SilenceAwareIDSConfig(
                state=self.SparseGPConfig(input_dim=1, inducing_points=4),
                observation=self.ObservationModelConfig(),
                missingness=missingness,
                state_training=self.StateTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    training_strategy="joint_variational",
                ),
                missingness_training=self.MissingnessTrainingConfig(epochs=1, batch_size=4),
                use_m2=True,
                use_m3=True,
                use_m5=False,
                homogeneous_missingness=False,
                sensor_conditional_missingness=True,
            )
        )
        summary = pipeline.fit(
            self.x_train,
            self.y_train,
            missing_indicator_train=self.missing_indicator,
            sensor_metadata_train=self.sensor_metadata,
        )
        self.assertIn("state_loss", summary.state_history)
        self.assertIn("joint_missingness_loss", summary.state_history)
        self.assertIsNotNone(summary.missingness_history)
        self.assertIn("health_kl_loss", summary.missingness_history)

    def test_infer_sensor_health_returns_latent_health_summary(self) -> None:
        pipeline = self._build_pipeline()
        pipeline.fit_state_model(self.x_train, self.y_train)
        health = pipeline.infer_sensor_health(
            self.x_eval,
            y=self.y_eval,
            sensor_metadata={
                "sensor_type": self.sensor_metadata["sensor_type"][: self.x_eval.shape[0]],
                "sensor_group": self.sensor_metadata["sensor_group"][: self.x_eval.shape[0]],
            },
            batch_size=2,
        )
        self.assertIsNotNone(health)
        self.assertIn("health_mean", health)
        self.assertIn("health_var", health)


if __name__ == "__main__":
    unittest.main()
