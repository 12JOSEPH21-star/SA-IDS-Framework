from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
GPYTORCH_AVAILABLE = importlib.util.find_spec("gpytorch") is not None

if TORCH_AVAILABLE:
    import torch


@unittest.skipUnless(TORCH_AVAILABLE and GPYTORCH_AVAILABLE, "requires torch and gpytorch")
class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        from experiment import ExperimentRunConfig, ResearchExperimentRunner, TabularDataConfig
        from models import MissingMechanismConfig, SparseGPConfig
        from pipeline import (
            MissingnessTrainingConfig,
            SilenceAwareIDSConfig,
            StateTrainingConfig,
        )

        self.ExperimentRunConfig = ExperimentRunConfig
        self.ResearchExperimentRunner = ResearchExperimentRunner
        self.TabularDataConfig = TabularDataConfig
        self.SilenceAwareIDSConfig = SilenceAwareIDSConfig
        self.SparseGPConfig = SparseGPConfig
        self.MissingMechanismConfig = MissingMechanismConfig
        self.StateTrainingConfig = StateTrainingConfig
        self.MissingnessTrainingConfig = MissingnessTrainingConfig

        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.csv_path = self.root / "study.csv"
        self._write_dataset(self.csv_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_dataset(self, path: Path) -> None:
        fieldnames = [
            "timestamp",
            "station_id",
            "latitude",
            "longitude",
            "cost",
            "temperature",
            "humidity",
            "pressure",
            "sensor_type",
            "sensor_group",
            "sensor_modality",
            "site_type",
            "maintenance_state",
            "maintenance_age",
        ]
        rows: list[dict[str, str]] = []
        timestamps = [
            "2025-01-01T00:00",
            "2025-01-01T01:00",
            "2025-01-01T02:00",
            "2025-01-01T03:00",
            "2025-01-01T04:00",
            "2025-01-01T05:00",
        ]
        stations = [
            ("S1", "35.10", "129.04", "1.2", "lidar", "remote", "optical", "coastal", "good", "0.5"),
            ("S2", "35.20", "128.90", "0.9", "aws", "surface", "mechanical", "urban", "worn", "2.0"),
        ]
        for t_index, timestamp in enumerate(timestamps):
            for s_index, station in enumerate(stations):
                station_id, lat, lon, cost, sensor_type, group, modality, site_type, maintenance, age = station
                target = "" if (t_index + s_index) % 5 == 0 else f"{14.0 + t_index + s_index * 0.3:.2f}"
                rows.append(
                    {
                        "timestamp": timestamp,
                        "station_id": station_id,
                        "latitude": lat,
                        "longitude": lon,
                        "cost": cost,
                        "temperature": target,
                        "humidity": f"{60 + t_index + s_index}",
                        "pressure": f"{1010 - t_index}",
                        "sensor_type": sensor_type,
                        "sensor_group": group,
                        "sensor_modality": modality,
                        "site_type": site_type,
                        "maintenance_state": maintenance,
                        "maintenance_age": age,
                    }
                )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _build_runner(self) -> object:
        data_config = self.TabularDataConfig(
            data_path=self.csv_path,
            context_columns=("humidity", "pressure"),
            continuous_metadata_columns=("maintenance_age",),
            sensor_type_column="sensor_type",
            sensor_group_column="sensor_group",
            sensor_modality_column="sensor_modality",
            installation_environment_column="site_type",
            maintenance_state_column="maintenance_state",
        )
        pipeline_config = self.SilenceAwareIDSConfig(
            state=self.SparseGPConfig(
                input_dim=7,
                inducing_points=4,
                spatial_dims=(0, 1),
                temporal_dims=(2, 3, 4, 5, 6),
            ),
            missingness=self.MissingMechanismConfig(include_s=True),
            state_training=self.StateTrainingConfig(epochs=1, batch_size=4),
            missingness_training=self.MissingnessTrainingConfig(epochs=1, batch_size=4),
            use_m2=True,
            use_m3=True,
            use_m5=False,
            homogeneous_missingness=False,
            sensor_conditional_missingness=True,
        )
        run_config = self.ExperimentRunConfig(
            variant_names=("base_gp_only", "gp_plus_sensor_conditional_missingness"),
            sensitivity_logit_scales=(0.5, 1.0),
            max_selections=2,
            benchmark_expansion_factor=2,
            prediction_batch_size=4,
        )
        return self.ResearchExperimentRunner(data_config, pipeline_config, run_config)

    def test_prepare_data_infers_metadata_cardinalities(self) -> None:
        runner = self._build_runner()
        prepared = runner.prepare_data()
        self.assertGreater(prepared.train.X.shape[0], 0)
        self.assertGreater(prepared.calibration.X.shape[0], 0)
        self.assertGreater(prepared.evaluation.X.shape[0], 0)
        self.assertEqual(prepared.metadata_cardinalities["sensor_type"], 2)
        self.assertEqual(prepared.metadata_cardinalities["sensor_group"], 2)

    def test_runner_produces_serializable_summary(self) -> None:
        runner = self._build_runner()
        result = runner.run()
        payload = result.to_dict()
        self.assertIn("base_metrics", payload)
        self.assertIn("ablations", payload)
        self.assertIn("dataset_summary", payload)
        self.assertIn("input_dim", payload["dataset_summary"])
        self.assertIn("runtime_environment", payload)
        self.assertIn("reproducibility", payload)
        self.assertIn("benchmark", payload)
        self.assertIn("predict_seconds", payload["benchmark"])

    def test_write_artifacts_exports_csv_and_report(self) -> None:
        runner = self._build_runner()
        artifacts = runner.write_artifacts(self.root / "outputs" / "summary.json")
        self.assertTrue(artifacts.summary_path.exists())
        self.assertTrue(artifacts.ablations_path.exists())
        self.assertTrue(artifacts.sensitivity_path.exists())
        self.assertTrue(artifacts.selection_path.exists())
        self.assertTrue(artifacts.report_path.exists())
        self.assertTrue(artifacts.progress_path is not None and artifacts.progress_path.exists())
        self.assertTrue(artifacts.checkpoint_path is not None and artifacts.checkpoint_path.exists())
        heartbeat_path = artifacts.summary_path.parent / "framework_heartbeat.jsonl"
        self.assertTrue(heartbeat_path.exists())
        self.assertIn('"stage": "evaluating_base_pipeline"', heartbeat_path.read_text(encoding="utf-8"))

        report_text = artifacts.report_path.read_text(encoding="utf-8")
        self.assertIn("Silence-Aware IDS Framework Report", report_text)
        self.assertIn("## Runtime", report_text)
        self.assertIn("## Benchmark", report_text)
        self.assertIn("gp_plus_sensor_conditional_missingness", report_text)
        progress_payload = json.loads(artifacts.progress_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        self.assertEqual(progress_payload["status"], "completed")
        self.assertEqual(progress_payload["stage"], "completed")

    def test_write_artifacts_writes_partial_summary_on_failure(self) -> None:
        runner = self._build_runner()
        output_path = self.root / "outputs" / "summary.json"

        with mock.patch.object(
            runner,
            "_run_ablation_suite_incremental",
            side_effect=RuntimeError("forced ablation failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced ablation failure"):
                runner.write_artifacts(output_path)

        self.assertTrue(output_path.exists())
        progress_path = output_path.parent / "framework_progress.json"
        self.assertTrue(progress_path.exists())
        summary_payload = json.loads(output_path.read_text(encoding="utf-8"))
        progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(summary_payload["status"], "failed")
        self.assertEqual(progress_payload["status"], "failed")
        self.assertIn("forced ablation failure", summary_payload["error"])

    def test_write_artifacts_resumes_from_completed_checkpoint(self) -> None:
        runner = self._build_runner()
        output_path = self.root / "outputs" / "summary.json"
        runner.write_artifacts(output_path)

        resumed_runner = self._build_runner()
        with mock.patch.object(resumed_runner, "run", side_effect=AssertionError("run() should not be called")):
            artifacts = resumed_runner.write_artifacts(output_path)

        self.assertTrue(artifacts.summary_path.exists())
        progress_payload = json.loads((output_path.parent / "framework_progress.json").read_text(encoding="utf-8"))
        self.assertEqual(progress_payload["status"], "completed")

    def test_large_run_auto_caps_rows_and_stabilizes_reliability_mode(self) -> None:
        runner = self._build_runner()
        prepared = runner.prepare_data()
        run_config = self.ExperimentRunConfig(
            max_train_rows=4,
            max_calibration_rows=2,
            max_evaluation_rows=2,
            auto_scale_large_runs=True,
        )
        runner = self.ResearchExperimentRunner(runner.data_config, runner.pipeline_config, run_config)
        capped, row_caps = runner._maybe_cap_prepared_data(prepared)
        resolved = runner._resolve_pipeline_config(capped)
        stabilized = runner._stabilize_pipeline_config_for_large_runs(resolved, row_caps=row_caps)
        self.assertTrue(row_caps["applied"])
        self.assertEqual(int(capped.train.X.shape[0]), 4)
        self.assertEqual(int(capped.calibration.X.shape[0]), 2)
        self.assertEqual(int(capped.evaluation.X.shape[0]), 2)
        self.assertEqual(stabilized.reliability.mode, "relational_adaptive")

    def test_large_run_uses_evaluation_proxy_for_full_batch_residency(self) -> None:
        runner = self._build_runner()
        prepared = runner.prepare_data()
        run_config = self.ExperimentRunConfig(
            max_train_rows=4,
            max_calibration_rows=2,
            max_evaluation_rows=2,
            use_evaluation_as_candidate_pool=True,
        )
        runner = self.ResearchExperimentRunner(runner.data_config, runner.pipeline_config, run_config)
        capped, row_caps = runner._maybe_cap_prepared_data(prepared)
        self.assertTrue(row_caps["applied"])
        self.assertEqual(row_caps["retained_full_source"], "evaluation_proxy")
        self.assertEqual(int(capped.full.X.shape[0]), int(capped.evaluation.X.shape[0]))

    def test_run_with_checkpoint_uses_prepared_cache(self) -> None:
        runner = self._build_runner()
        output_path = self.root / "outputs" / "summary.json"
        prepared = runner.prepare_data()
        capped, row_caps = runner._maybe_cap_prepared_data(prepared)
        cache_path = runner._prepared_cache_path(output_path)
        runner._save_prepared_cache(cache_path, capped, row_caps)

        with mock.patch.object(runner, "prepare_data", side_effect=AssertionError("prepare_data should not be called")):
            result, progress_path, checkpoint_path = runner._run_with_checkpoint(output_path)

        self.assertTrue(progress_path.exists())
        self.assertTrue(checkpoint_path.exists())
        self.assertIn("rmse", result.base_metrics)

    def test_policy_ablation_candidate_pool_is_capped(self) -> None:
        runner = self._build_runner()
        prepared = runner.prepare_data()
        resolved = runner._resolve_pipeline_config(prepared)
        stabilized = runner._stabilize_pipeline_config_for_large_runs(
            resolved,
            row_caps={"applied": True},
        )
        from pipeline import SilenceAwareIDS

        pipeline = SilenceAwareIDS(stabilized)
        variant = pipeline.spawn_ablation_variant("full_model")
        candidate_rows = 6000
        candidate_x = prepared.evaluation.X.repeat((candidate_rows // prepared.evaluation.X.shape[0]) + 1, 1)[:candidate_rows]
        candidate_cost = prepared.evaluation.cost.repeat((candidate_rows // prepared.evaluation.cost.shape[0]) + 1)[:candidate_rows]
        candidate_context = prepared.evaluation.context.repeat((candidate_rows // prepared.evaluation.context.shape[0]) + 1, 1)[:candidate_rows]
        candidate_m = prepared.evaluation.M.repeat((candidate_rows // prepared.evaluation.M.shape[0]) + 1)[:candidate_rows]
        candidate_s = prepared.evaluation.S.repeat((candidate_rows // prepared.evaluation.S.shape[0]) + 1)[:candidate_rows]
        metadata = prepared.evaluation.sensor_metadata
        repeats = (candidate_rows // metadata.sensor_type.shape[0]) + 1
        candidate_metadata = type(metadata)(
            sensor_instance=metadata.sensor_instance.repeat(repeats)[:candidate_rows],
            sensor_type=metadata.sensor_type.repeat(repeats)[:candidate_rows],
            sensor_group=metadata.sensor_group.repeat(repeats)[:candidate_rows],
            sensor_modality=metadata.sensor_modality.repeat(repeats)[:candidate_rows],
            installation_environment=metadata.installation_environment.repeat(repeats)[:candidate_rows],
            maintenance_state=metadata.maintenance_state.repeat(repeats)[:candidate_rows],
            continuous=metadata.continuous.repeat(repeats, 1)[:candidate_rows],
        )

        capped = runner._cap_candidate_pool(
            variant,
            candidate_x=candidate_x,
            candidate_cost=candidate_cost,
            candidate_context=candidate_context,
            candidate_M=candidate_m,
            candidate_S=candidate_s,
            candidate_sensor_metadata=candidate_metadata,
        )
        capped_x, capped_cost, capped_context, capped_m, capped_s, capped_metadata, info = capped
        self.assertTrue(info["applied"])
        self.assertLess(int(capped_x.shape[0]), candidate_rows)
        self.assertEqual(int(capped_x.shape[0]), int(capped_cost.shape[0]))
        self.assertEqual(int(capped_x.shape[0]), int(capped_context.shape[0]))
        self.assertEqual(int(capped_x.shape[0]), int(capped_m.shape[0]))
        self.assertEqual(int(capped_x.shape[0]), int(capped_s.shape[0]))
        self.assertEqual(int(capped_x.shape[0]), int(capped_metadata.sensor_type.shape[0]))

    def test_policy_ablation_evaluation_rows_are_capped(self) -> None:
        runner = self._build_runner()
        prepared = runner.prepare_data()
        resolved = runner._resolve_pipeline_config(prepared)
        stabilized = runner._stabilize_pipeline_config_for_large_runs(
            resolved,
            row_caps={"applied": True},
        )
        from pipeline import SilenceAwareIDS

        pipeline = SilenceAwareIDS(stabilized)
        variant = pipeline.spawn_ablation_variant("full_model")
        eval_rows = 6000
        eval_x = prepared.evaluation.X.repeat((eval_rows // prepared.evaluation.X.shape[0]) + 1, 1)[:eval_rows]
        eval_y = prepared.evaluation.y.repeat((eval_rows // prepared.evaluation.y.shape[0]) + 1)[:eval_rows]
        eval_context = prepared.evaluation.context.repeat((eval_rows // prepared.evaluation.context.shape[0]) + 1, 1)[:eval_rows]
        eval_m = prepared.evaluation.M.repeat((eval_rows // prepared.evaluation.M.shape[0]) + 1)[:eval_rows]
        eval_s = prepared.evaluation.S.repeat((eval_rows // prepared.evaluation.S.shape[0]) + 1)[:eval_rows]
        metadata = prepared.evaluation.sensor_metadata
        repeats = (eval_rows // metadata.sensor_type.shape[0]) + 1
        eval_metadata = type(metadata)(
            sensor_instance=metadata.sensor_instance.repeat(repeats)[:eval_rows],
            sensor_type=metadata.sensor_type.repeat(repeats)[:eval_rows],
            sensor_group=metadata.sensor_group.repeat(repeats)[:eval_rows],
            sensor_modality=metadata.sensor_modality.repeat(repeats)[:eval_rows],
            installation_environment=metadata.installation_environment.repeat(repeats)[:eval_rows],
            maintenance_state=metadata.maintenance_state.repeat(repeats)[:eval_rows],
            continuous=metadata.continuous.repeat(repeats, 1)[:eval_rows],
        )

        capped = runner._cap_ablation_evaluation(
            variant,
            X_eval=eval_x,
            y_eval=eval_y,
            context_eval=eval_context,
            M_eval=eval_m,
            S_eval=eval_s,
            sensor_metadata_eval=eval_metadata,
            batch_size=128,
        )
        capped_x, capped_y, capped_context, capped_m, capped_s, capped_metadata, capped_batch_size, info = capped
        self.assertTrue(info["applied"])
        self.assertLess(int(capped_x.shape[0]), eval_rows)
        self.assertEqual(int(capped_x.shape[0]), int(capped_y.shape[0]))
        self.assertEqual(int(capped_x.shape[0]), int(capped_context.shape[0]))
        self.assertEqual(int(capped_x.shape[0]), int(capped_m.shape[0]))
        self.assertEqual(int(capped_x.shape[0]), int(capped_s.shape[0]))
        self.assertEqual(int(capped_x.shape[0]), int(capped_metadata.sensor_type.shape[0]))
        self.assertEqual(int(capped_batch_size), 64)

    def test_ensure_dynamic_silence_recomputes_only_when_needed(self) -> None:
        runner = self._build_runner()
        prepared = runner.prepare_data()
        resolved = runner._resolve_pipeline_config(prepared)
        from pipeline import SilenceAwareIDS

        pipeline = SilenceAwareIDS(resolved)
        heartbeat_path = self.root / "heartbeat.jsonl"
        self.assertEqual(int(prepared.evaluation.S.sum().item()), 0)

        with mock.patch.object(
            pipeline,
            "detect_silence",
            return_value={"dynamic_silence": prepared.evaluation.missing_indicator.clone()},
        ) as detect_mock:
            ensured = runner._ensure_dynamic_silence(
                pipeline,
                prepared.evaluation,
                batch_size=4,
                heartbeat_path=heartbeat_path,
                stage="running_sensitivity",
            )

        detect_mock.assert_called_once()
        self.assertIsNotNone(ensured)
        self.assertGreater(int(ensured.sum().item()), 0)
        heartbeat_text = heartbeat_path.read_text(encoding="utf-8")
        self.assertIn('"stage": "running_sensitivity"', heartbeat_text)
        self.assertIn('"step": "evaluation_silence_complete"', heartbeat_text)

    def test_sensitivity_rows_are_capped_for_large_runs(self) -> None:
        runner = self._build_runner()
        prepared = runner.prepare_data()
        large_rows = 10000
        repeated_eval = self.ResearchExperimentRunner._slice_batch_rows(
            prepared.evaluation,
            torch.arange(prepared.evaluation.X.shape[0], device=prepared.evaluation.X.device).repeat(
                (large_rows // prepared.evaluation.X.shape[0]) + 1
            )[:large_rows],
        )
        prepared_large = type(prepared)(
            full=repeated_eval,
            train=prepared.train,
            calibration=prepared.calibration,
            evaluation=repeated_eval,
            metadata_cardinalities=prepared.metadata_cardinalities,
            feature_schema=prepared.feature_schema,
        )
        capped = runner._cap_sensitivity_evaluation(prepared_large, batch_size=512)
        capped_x, capped_y, capped_context, capped_m, capped_s, capped_metadata, capped_batch_size, info = capped
        self.assertTrue(info["applied"])
        self.assertEqual(int(capped_x.shape[0]), 8192)
        self.assertEqual(int(capped_x.shape[0]), int(capped_y.shape[0]))
        self.assertEqual(int(capped_x.shape[0]), int(capped_context.shape[0]))
        self.assertEqual(int(capped_x.shape[0]), int(capped_m.shape[0]))
        self.assertEqual(int(capped_x.shape[0]), int(capped_s.shape[0]))
        self.assertEqual(int(capped_x.shape[0]), int(capped_metadata.sensor_type.shape[0]))
        self.assertEqual(int(capped_batch_size), 256)

    def test_benchmark_candidate_limit_is_stabilized_for_large_runs(self) -> None:
        runner = self._build_runner()
        prepared = runner.prepare_data()
        large_rows = 32768
        repeated_eval = self.ResearchExperimentRunner._slice_batch_rows(
            prepared.evaluation,
            torch.arange(prepared.evaluation.X.shape[0], device=prepared.evaluation.X.device).repeat(
                (large_rows // prepared.evaluation.X.shape[0]) + 1
            )[:large_rows],
        )
        prepared_large = type(prepared)(
            full=repeated_eval,
            train=prepared.train,
            calibration=prepared.calibration,
            evaluation=repeated_eval,
            metadata_cardinalities=prepared.metadata_cardinalities,
            feature_schema=prepared.feature_schema,
        )
        self.assertEqual(runner._effective_benchmark_candidate_limit(prepared_large), 65536)


if __name__ == "__main__":
    unittest.main()
