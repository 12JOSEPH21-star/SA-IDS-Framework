import io
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from task_cli.app import main


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
GPYTORCH_AVAILABLE = importlib.util.find_spec("gpytorch") is not None


class ResearchCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config_path = self.root / "research_config.json"
        self.data_path = self.root / "sample_weather.csv"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_init_creates_template_files(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["init", "--config", str(self.config_path), "--data", str(self.data_path)])

        self.assertEqual(code, 0)
        self.assertTrue(self.config_path.exists())
        self.assertTrue(self.data_path.exists())

        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["target_column"], "temperature")
        self.assertEqual(payload["removal_ratios"], [0.1, 0.3, 0.5, 0.7])
        self.assertEqual(payload["sparse_focus_min_ratio"], 0.5)
        self.assertEqual(payload["conformal_alpha"], 0.1)
        self.assertIn("Created config", output.getvalue())

    def test_run_generates_study_outputs(self) -> None:
        self.assertEqual(main(["init", "--config", str(self.config_path), "--data", str(self.data_path)]), 0)

        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        payload["output_dir"] = "outputs/test_run"
        self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["run", "--config", str(self.config_path)])

        self.assertEqual(code, 0)
        study_dir = self.root / "outputs" / "test_run"
        for filename in [
            "metrics.csv",
            "summary.json",
            "tradeoff.svg",
            "report.md",
            "station_ranking.csv",
            "test_predictions.csv",
            "coefficients.csv",
            "policy_priority.csv",
            "condition_metrics.csv",
            "policy_validation.csv",
        ]:
            self.assertTrue((study_dir / filename).exists(), filename)

        metrics_text = (study_dir / "metrics.csv").read_text(encoding="utf-8")
        self.assertIn("Baseline-X", metrics_text)
        self.assertIn("Proposed-AB", metrics_text)

        summary = json.loads((study_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("recommendation", summary)
        self.assertIn("sparse_scenarios", summary)
        self.assertIn("hypothesis_checks", summary)
        self.assertIn("reliability_policy", summary)
        self.assertIn("mechanism_summary", summary)
        self.assertIn("policy_validation", summary)
        self.assertIn("ablation", summary)
        self.assertTrue(any(item["scenario"] == "Sparse-50" for item in summary["sparse_scenarios"]))
        self.assertTrue(any(item["variant"] == "Proposed-Full" for item in summary["metrics"]))
        self.assertIn("conformal_target_coverage", summary["reliability_policy"])

        report_text = (study_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("Sparse Scenario Summary", report_text)
        self.assertIn("Policy-Level Validation", report_text)
        self.assertIn("Why The Full Method Is Needed", report_text)
        self.assertIn("H5", report_text)
        self.assertIn("Coverage@90", report_text)
        self.assertIn("Study finished", output.getvalue())

    def test_run_returns_error_for_missing_columns(self) -> None:
        bad_data = self.root / "bad.csv"
        bad_data.write_text("timestamp,station_id,temperature\n2025-01-01T00:00,S1,1.0\n", encoding="utf-8")
        payload = {
            "data_path": str(bad_data),
            "output_dir": "outputs/bad_run",
        }
        self.config_path.write_text(json.dumps(payload), encoding="utf-8")

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["run", "--config", str(self.config_path)])

        self.assertEqual(code, 1)
        self.assertIn("missing required columns", output.getvalue())

    def test_framework_init_creates_template_files(self) -> None:
        framework_config = self.root / "framework_config.json"
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["framework-init", "--config", str(framework_config), "--data", str(self.data_path)])

        self.assertEqual(code, 0)
        self.assertTrue(framework_config.exists())
        payload = json.loads(framework_config.read_text(encoding="utf-8"))
        self.assertIn("data", payload)
        self.assertIn("pipeline", payload)
        self.assertIn("run", payload)
        self.assertEqual(payload["data"]["sensor_type_column"], "sensor_type")
        self.assertEqual(payload["data"]["maintenance_state_column"], "maintenance_state")
        self.assertIn("gp_plus_dynamic_silence", payload["run"]["variant_names"])
        self.assertIn("gp_plus_joint_variational_missingness", payload["run"]["variant_names"])
        self.assertIn("gp_plus_joint_jvi_training", payload["run"]["variant_names"])
        self.assertIn("gp_plus_pattern_mixture_missingness", payload["run"]["variant_names"])
        self.assertIn("gp_plus_conformal_reliability", payload["run"]["variant_names"])
        self.assertIn("relational_reliability_baseline", payload["run"]["variant_names"])
        self.assertIn("myopic_policy_baseline", payload["run"]["variant_names"])
        self.assertIn("ppo_warmstart_baseline", payload["run"]["variant_names"])
        self.assertIn("rollout_policy_baseline", payload["run"]["variant_names"])
        self.assertEqual(payload["pipeline"]["missingness"]["inference_strategy"], "joint_variational")
        self.assertEqual(payload["pipeline"]["state_training"]["training_strategy"], "joint_variational")
        self.assertEqual(payload["pipeline"]["policy"]["planning_strategy"], "ppo_online")
        self.assertTrue(payload["pipeline"]["observation"]["use_pi_ssd"])
        self.assertTrue(payload["pipeline"]["observation"]["use_dbn"])
        self.assertTrue(payload["pipeline"]["observation"]["use_latent_ode"])
        self.assertEqual(payload["pipeline"]["observation"]["corruption_probability_start"], 0.05)
        self.assertEqual(payload["pipeline"]["observation"]["corruption_probability_end"], 0.2)
        self.assertTrue(payload["pipeline"]["missingness"]["use_sensor_health_latent"])
        self.assertEqual(payload["pipeline"]["reliability"]["mode"], "graph_corel")
        self.assertEqual(payload["pipeline"]["reliability"]["graph_message_passing_steps"], 2)
        self.assertEqual(payload["run"]["seed"], 7)
        self.assertEqual(payload["run"]["benchmark_expansion_factor"], 4)
        self.assertIn("Created framework config", output.getvalue())

    def test_framework_init_supports_real_data_preset(self) -> None:
        framework_config = self.root / "framework_isd.json"
        isd_path = self.root / "isd_hourly.csv"
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "framework-init",
                    "--config",
                    str(framework_config),
                    "--data",
                    str(isd_path),
                    "--preset",
                    "isd_hourly",
                ]
            )

        self.assertEqual(code, 0)
        payload = json.loads(framework_config.read_text(encoding="utf-8"))
        self.assertEqual(payload["preset"], "isd_hourly")
        self.assertEqual(payload["data"]["target_column"], "air_temperature")
        self.assertEqual(payload["data"]["cost_column"], "sensor_cost")
        self.assertIn("Configured dataset path", output.getvalue())
        self.assertFalse(isd_path.exists())

    @unittest.skipIf(TORCH_AVAILABLE and GPYTORCH_AVAILABLE, "only checks missing-dependency fallback")
    def test_framework_run_reports_missing_dependencies(self) -> None:
        framework_config = self.root / "framework_config.json"
        self.assertEqual(
            main(["framework-init", "--config", str(framework_config), "--data", str(self.data_path)]),
            0,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["framework-run", "--config", str(framework_config)])

        self.assertEqual(code, 1)
        self.assertIn("requirements-research.txt", output.getvalue())

    @unittest.skipUnless(TORCH_AVAILABLE and GPYTORCH_AVAILABLE, "requires torch and gpytorch")
    def test_framework_run_writes_framework_artifacts(self) -> None:
        framework_config = self.root / "framework_config.json"
        self.assertEqual(
            main(["framework-init", "--config", str(framework_config), "--data", str(self.data_path)]),
            0,
        )
        payload = json.loads(framework_config.read_text(encoding="utf-8"))
        payload["pipeline"]["state_training"]["epochs"] = 1
        payload["pipeline"]["missingness_training"]["epochs"] = 1
        payload["run"]["variant_names"] = [
            "base_gp_only",
            "gp_plus_sensor_conditional_missingness",
        ]
        payload["run"]["sensitivity_logit_scales"] = [0.5, 1.0]
        payload["output_path"] = "outputs/framework_test/summary.json"
        framework_config.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["framework-run", "--config", str(framework_config)])

        self.assertEqual(code, 0)
        run_dir = self.root / "outputs" / "framework_test"
        for filename in ["summary.json", "ablations.csv", "sensitivity.csv", "selection.csv", "report.md"]:
            self.assertTrue((run_dir / filename).exists(), filename)

        summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("runtime_environment", summary_payload)
        self.assertIn("reproducibility", summary_payload)
        self.assertIn("benchmark", summary_payload)
        self.assertIn("predict_seconds", summary_payload["benchmark"])
        self.assertIn("input_tensor_bytes", summary_payload["benchmark"])
        report_text = (run_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("Silence-Aware IDS Framework Report", report_text)
        self.assertIn("## Runtime", report_text)
        self.assertIn("## Benchmark", report_text)
        self.assertIn("gp_plus_sensor_conditional_missingness", report_text)
        self.assertIn("Ablations:", output.getvalue())
        self.assertIn("Report:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
