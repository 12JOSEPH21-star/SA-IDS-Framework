from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
GPYTORCH_AVAILABLE = importlib.util.find_spec("gpytorch") is not None


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

        report_text = artifacts.report_path.read_text(encoding="utf-8")
        self.assertIn("Silence-Aware IDS Framework Report", report_text)
        self.assertIn("## Runtime", report_text)
        self.assertIn("## Benchmark", report_text)
        self.assertIn("gp_plus_sensor_conditional_missingness", report_text)


if __name__ == "__main__":
    unittest.main()
