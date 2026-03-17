from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from benchmark_suite import BenchmarkSuiteConfig, BenchmarkSuiteRunner

from task_cli.app import main


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
GPYTORCH_AVAILABLE = importlib.util.find_spec("gpytorch") is not None


class BenchmarkSuiteCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.framework_config = self.root / "framework_config.json"
        self.data_path = self.root / "sample_weather.csv"
        self.benchmark_config = self.root / "benchmark_config.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_benchmark_init_writes_config(self) -> None:
        self.assertEqual(
            main(["framework-init", "--config", str(self.framework_config), "--data", str(self.data_path)]),
            0,
        )
        code = main(
            [
                "benchmark-init",
                "--config",
                str(self.benchmark_config),
                "--framework-config",
                str(self.framework_config),
            ]
        )
        self.assertEqual(code, 0)
        payload = json.loads(self.benchmark_config.read_text(encoding="utf-8"))
        self.assertEqual(Path(payload["framework_config_path"]).resolve(), self.framework_config.resolve())
        self.assertEqual(payload["preset"], "small")
        self.assertEqual(payload["temporal_stride_hours"], 6)
        self.assertEqual(payload["missing_intensities"], [0.3, 0.5, 0.7])
        self.assertTrue(payload["cache_prepared_data"])
        self.assertEqual(payload["repeat_seeds"], [7, 11, 19])
        self.assertEqual(payload["bootstrap_samples"], 256)

    def test_benchmark_init_supports_medium_preset(self) -> None:
        self.assertEqual(
            main(["framework-init", "--config", str(self.framework_config), "--data", str(self.data_path)]),
            0,
        )
        code = main(
            [
                "benchmark-init",
                "--config",
                str(self.benchmark_config),
                "--framework-config",
                str(self.framework_config),
                "--preset",
                "medium",
            ]
        )
        self.assertEqual(code, 0)
        payload = json.loads(self.benchmark_config.read_text(encoding="utf-8"))
        self.assertEqual(payload["preset"], "medium")
        self.assertEqual(payload["state_epochs"], 2)
        self.assertEqual(payload["station_limit"], 16)
        self.assertEqual(payload["repeat_seeds"], [7, 11, 19, 23, 29])

    def test_benchmark_init_supports_large_preset(self) -> None:
        self.assertEqual(
            main(["framework-init", "--config", str(self.framework_config), "--data", str(self.data_path)]),
            0,
        )
        code = main(
            [
                "benchmark-init",
                "--config",
                str(self.benchmark_config),
                "--framework-config",
                str(self.framework_config),
                "--preset",
                "large",
            ]
        )
        self.assertEqual(code, 0)
        payload = json.loads(self.benchmark_config.read_text(encoding="utf-8"))
        self.assertEqual(payload["preset"], "large")
        self.assertEqual(payload["station_limit"], 96)
        self.assertEqual(payload["prediction_batch_size"], 512)
        self.assertEqual(
            payload["region_holdout_variants"],
            ["base_gp_only", "gp_plus_joint_generative_jvi_training", "full_model"],
        )

    @unittest.skipUnless(TORCH_AVAILABLE and GPYTORCH_AVAILABLE, "requires torch and gpytorch")
    def test_full_model_policy_override_uses_budget_mode(self) -> None:
        self.assertEqual(
            main(["framework-init", "--config", str(self.framework_config), "--data", str(self.data_path)]),
            0,
        )
        runner = BenchmarkSuiteRunner(
            BenchmarkSuiteConfig(
                framework_config_path=self.framework_config,
                output_dir=self.root / "outputs" / "benchmark_suite",
            )
        )
        variant = runner._variant_config("full_model")
        self.assertEqual(variant.policy.selection_mode, "budget")
        self.assertGreaterEqual(variant.policy.route_distance_weight, 0.2)
        self.assertGreaterEqual(variant.reliability.graph_min_quantile_factor, 0.95)
        self.assertLessEqual(variant.reliability.graph_score_weight, 0.08)

    @unittest.skipUnless(TORCH_AVAILABLE and GPYTORCH_AVAILABLE, "requires torch and gpytorch")
    def test_medium_full_model_uses_relational_reliability(self) -> None:
        self.assertEqual(
            main(["framework-init", "--config", str(self.framework_config), "--data", str(self.data_path)]),
            0,
        )
        runner = BenchmarkSuiteRunner(
            BenchmarkSuiteConfig(
                framework_config_path=self.framework_config,
                output_dir=self.root / "outputs" / "benchmark_suite",
                max_train_rows=8192,
                max_calibration_rows=2048,
                max_evaluation_rows=2048,
            )
        )
        variant = runner._variant_config("full_model")
        self.assertEqual(variant.reliability.mode, "relational_adaptive")

    @unittest.skipUnless(TORCH_AVAILABLE and GPYTORCH_AVAILABLE, "requires torch and gpytorch")
    def test_random_policy_selection_is_seeded(self) -> None:
        import torch

        self.assertEqual(
            main(["framework-init", "--config", str(self.framework_config), "--data", str(self.data_path)]),
            0,
        )
        runner = BenchmarkSuiteRunner(
            BenchmarkSuiteConfig(
                framework_config_path=self.framework_config,
                output_dir=self.root / "outputs" / "benchmark_suite",
            )
        )
        candidate_cost = torch.ones(8)
        first = runner._random_selection(candidate_cost, budget=5.0, max_selections=5)
        second = runner._random_selection(candidate_cost, budget=5.0, max_selections=5)
        self.assertEqual(first["selected_indices"], second["selected_indices"])

    @unittest.skipUnless(TORCH_AVAILABLE and GPYTORCH_AVAILABLE, "requires torch and gpytorch")
    def test_fault_variants_use_conservative_false_alarm_weights(self) -> None:
        self.assertEqual(
            main(["framework-init", "--config", str(self.framework_config), "--data", str(self.data_path)]),
            0,
        )
        runner = BenchmarkSuiteRunner(
            BenchmarkSuiteConfig(
                framework_config_path=self.framework_config,
                output_dir=self.root / "outputs" / "benchmark_suite",
            )
        )
        variant = runner._fault_variant_config("dbn_plus_pi_ssd")
        self.assertLessEqual(variant.observation.fault_score_probability_weight, 0.3)
        self.assertGreaterEqual(variant.observation.fault_score_temporal_weight, 0.7)
        self.assertLessEqual(variant.observation.fault_target_false_alarm_rate, 0.05)

    @unittest.skipUnless(TORCH_AVAILABLE and GPYTORCH_AVAILABLE, "requires torch and gpytorch")
    def test_benchmark_run_writes_artifacts(self) -> None:
        self.assertEqual(
            main(["framework-init", "--config", str(self.framework_config), "--data", str(self.data_path)]),
            0,
        )
        self.assertEqual(
            main(
                [
                    "benchmark-init",
                    "--config",
                    str(self.benchmark_config),
                    "--framework-config",
                    str(self.framework_config),
                ]
            ),
            0,
        )
        payload = json.loads(self.benchmark_config.read_text(encoding="utf-8"))
        payload["output_dir"] = str((self.root / "outputs" / "benchmark_suite").resolve())
        payload["station_limit"] = 4
        payload["temporal_stride_hours"] = 3
        payload["max_train_rows"] = 128
        payload["max_calibration_rows"] = 48
        payload["max_evaluation_rows"] = 48
        payload["state_epochs"] = 1
        payload["missingness_epochs"] = 1
        payload["inducing_points"] = 8
        payload["prediction_batch_size"] = 32
        payload["missing_intensities"] = [0.3]
        payload["predictive_mechanisms"] = ["mar"]
        payload["fault_scenarios"] = ["random_dropout", "spike_burst"]
        payload["repeat_seeds"] = [7]
        self.benchmark_config.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        code = main(["benchmark-run", "--config", str(self.benchmark_config)])
        self.assertEqual(code, 0)
        output_dir = self.root / "outputs" / "benchmark_suite"
        for filename in [
            "summary.json",
            "predictive_mnar.csv",
            "fault_diagnosis.csv",
            "reliability_shift.csv",
            "coverage_over_time.csv",
            "region_holdout.csv",
            "ablation.csv",
            "policy_runtime.csv",
            "runtime.csv",
            "significance_summary.json",
            "paper_tables.json",
            "report.md",
        ]:
            self.assertTrue((output_dir / filename).exists(), filename)
        self.assertTrue((output_dir / "figures" / "table_1_predictive_mnar.csv").exists())
        self.assertTrue((output_dir / "figures" / "table_1_predictive_significance.csv").exists())
        self.assertTrue((output_dir / "figures" / "figure_2_coverage_over_time.csv").exists())
        paper_tables = json.loads((output_dir / "paper_tables.json").read_text(encoding="utf-8"))
        self.assertIn("total_operational_cost_mean", paper_tables["table_3_active_sensing"][0])
        self.assertIn("significance", paper_tables)


if __name__ == "__main__":
    unittest.main()
