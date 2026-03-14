from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiment import ExperimentArtifacts


def available_framework_presets() -> tuple[str, ...]:
    """Return supported framework configuration presets."""
    return ("demo", "isd_hourly", "aws_network", "era5_reference")


def _preset_payload(preset: str, data_path: Path, *, relative_to: Path) -> dict[str, object]:
    """Build a framework config payload for one schema preset."""
    from .research import _portable_path

    if preset == "isd_hourly":
        data_block = {
            "data_path": _portable_path(data_path, relative_to=relative_to),
            "timestamp_column": "timestamp",
            "station_id_column": "station_id",
            "target_column": "air_temperature",
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "context_columns": ["dew_point_temperature", "sea_level_pressure", "wind_speed"],
            "continuous_metadata_columns": ["elevation", "station_age_years"],
            "cost_column": "sensor_cost",
            "sensor_type_column": "sensor_type",
            "sensor_group_column": "sensor_group",
            "sensor_modality_column": "sensor_modality",
            "installation_environment_column": "site_type",
            "maintenance_state_column": "maintenance_state",
            "train_ratio": 0.7,
            "calibration_ratio": 0.15,
            "split_strategy": "temporal",
        }
    elif preset == "aws_network":
        data_block = {
            "data_path": _portable_path(data_path, relative_to=relative_to),
            "timestamp_column": "timestamp",
            "station_id_column": "station_id",
            "target_column": "temperature",
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "context_columns": ["humidity", "pressure", "wind_speed"],
            "continuous_metadata_columns": ["elevation", "maintenance_age"],
            "cost_column": "cost",
            "sensor_type_column": "sensor_type",
            "sensor_group_column": "sensor_group",
            "sensor_modality_column": "sensor_modality",
            "installation_environment_column": "site_type",
            "maintenance_state_column": "maintenance_state",
            "train_ratio": 0.7,
            "calibration_ratio": 0.15,
            "split_strategy": "temporal",
        }
    elif preset == "era5_reference":
        data_block = {
            "data_path": _portable_path(data_path, relative_to=relative_to),
            "timestamp_column": "time",
            "station_id_column": "grid_id",
            "target_column": "t2m",
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "context_columns": ["u10", "v10", "sp", "tp"],
            "continuous_metadata_columns": ["orography", "land_sea_mask"],
            "cost_column": None,
            "sensor_type_column": None,
            "sensor_group_column": None,
            "sensor_modality_column": None,
            "installation_environment_column": None,
            "maintenance_state_column": None,
            "train_ratio": 0.7,
            "calibration_ratio": 0.15,
            "split_strategy": "temporal",
        }
    else:
        data_block = {
            "data_path": _portable_path(data_path, relative_to=relative_to),
            "timestamp_column": "timestamp",
            "station_id_column": "station_id",
            "target_column": "temperature",
            "latitude_column": "latitude",
            "longitude_column": "longitude",
            "context_columns": ["humidity", "pressure"],
            "continuous_metadata_columns": ["elevation", "maintenance_age"],
            "cost_column": "cost",
            "sensor_type_column": "sensor_type",
            "sensor_group_column": "sensor_group",
            "sensor_modality_column": "sensor_modality",
            "installation_environment_column": "site_type",
            "maintenance_state_column": "maintenance_state",
            "train_ratio": 0.7,
            "calibration_ratio": 0.15,
            "split_strategy": "temporal",
        }

    payload = {
        "preset": preset,
        "data": data_block,
        "pipeline": {
            "state": {
                "input_dim": 7,
                "inducing_points": 32,
                "spatial_dims": [0, 1],
                "temporal_dims": [2, 3, 4, 5, 6],
            },
            "observation": {
                "threshold_mode": "hybrid" if preset == "era5_reference" else "stddev",
                "diagnosis_mode": "temporal",
                "use_pi_ssd": True,
                "use_dbn": True,
                "use_latent_ode": True,
                "stddev_multiplier": 2.5,
                "quantile_level": 0.95,
                "hybrid_floor": 0.5,
                "corruption_probability_start": 0.05,
                "corruption_probability_end": 0.2,
                "corruption_noise_std": 0.05,
                "latent_ode_weight": 0.05,
                "use_observation_noise": True,
                "require_calibration_split_for_quantile": True,
            },
            "missingness": {
                "mode": "selection",
                "inference_strategy": "joint_variational",
                "include_s": True,
                "include_dynamic_residual": True,
                "include_dynamic_threshold": True,
                "include_normalized_residual": True,
                "include_dynamic_feature_availability": True,
                "use_dynamic_features_for_training": True,
                "use_dynamic_features_for_prediction": True,
                "use_dynamic_features_for_policy": False,
                "use_sensor_health_latent": True,
                "hidden_dim": 32,
                "trunk_depth": 2,
                "encoder_hidden_dim": 32,
                "encoder_depth": 2,
                "reconstruction_weight": 1.0,
                "kl_weight": 0.01,
                "health_kl_weight": 0.01,
                "health_reconstruction_weight": 0.1,
            },
            "policy": {
                "utility_surrogate": "mi_proxy",
                "planning_strategy": "ppo_online",
                "selection_mode": "budget",
                "spatial_distance_dims": [0, 1],
                "observation_noise": 0.05,
                "redundancy_strength": 0.95,
                "planning_horizon": 3,
                "future_discount": 0.8,
                "lookahead_strength": 0.5,
                "ppo_epochs": 10,
                "ppo_policy_weight": 0.25,
                "ppo_max_candidates": 1024,
            },
            "reliability": {
                "prediction_target": "observation",
                "mode": "graph_corel",
                "adaptation_rate": 0.02,
                "relational_neighbor_weight": 0.25,
                "relational_temperature": 1.0,
                "graph_k_neighbors": 8,
                "graph_message_passing_steps": 2,
                "graph_training_steps": 50,
                "graph_score_weight": 0.5,
                "graph_covariance_weight": 0.5,
            },
            "state_training": {
                "epochs": 5,
                "training_strategy": "joint_variational",
                "batch_size": 64,
                "learning_rate": 0.01,
                "joint_missingness_weight": 1.0,
            },
            "missingness_training": {
                "epochs": 5,
                "batch_size": 128,
                "learning_rate": 0.001,
            },
            "use_m2": True,
            "use_m3": preset != "era5_reference",
            "use_m5": True,
            "homogeneous_missingness": False,
            "sensor_conditional_missingness": preset != "era5_reference",
        },
        "run": {
            "variant_names": [
                "base_gp_only",
                "gp_plus_dynamic_silence",
                "gp_plus_homogeneous_missingness",
                "gp_plus_sensor_conditional_missingness",
                "gp_plus_joint_variational_missingness",
                "gp_plus_joint_jvi_training",
                "gp_plus_pattern_mixture_missingness",
                "gp_plus_conformal_reliability",
                "relational_reliability_baseline",
                "myopic_policy_baseline",
                "ppo_warmstart_baseline",
                "rollout_policy_baseline",
                "variance_policy_baseline",
                "full_model",
            ],
            "sensitivity_logit_scales": [0.5, 1.0, 1.5, 2.0],
            "max_selections": 3,
            "seed": 7,
            "deterministic_algorithms": False,
            "torch_num_threads": 4,
            "matmul_precision": "high",
            "prediction_batch_size": 512,
            "benchmark_expansion_factor": 4,
        },
        "output_path": "outputs/framework_run/summary.json",
    }
    if preset == "era5_reference":
        payload["run"]["variant_names"] = [
            "base_gp_only",
            "gp_plus_dynamic_silence",
            "gp_plus_conformal_reliability",
        ]
    return payload


def write_framework_template(
    config_path: Path,
    data_path: Path,
    *,
    preset: str = "demo",
    force: bool = False,
) -> tuple[Path, Path]:
    """Write a ready-to-run research framework config and optional demo CSV.

    Args:
        config_path: Output JSON config path.
        data_path: Output CSV dataset path.
        preset: Schema preset name.
        force: Whether to overwrite existing files.

    Returns:
        Paths to the written config and dataset.
    """
    from .research import generate_demo_dataset

    if preset not in available_framework_presets():
        raise ValueError(f"preset must be one of {available_framework_presets()}.")
    if config_path.exists() and not force:
        raise FileExistsError(f"{config_path} already exists. Re-run with --force to overwrite.")
    if preset == "demo" and data_path.exists() and not force:
        raise FileExistsError(f"{data_path} already exists. Re-run with --force to overwrite.")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    if preset == "demo":
        data_path.parent.mkdir(parents=True, exist_ok=True)
        generate_demo_dataset(data_path)
    payload = _preset_payload(preset, data_path, relative_to=config_path.parent)
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path, data_path


def run_framework(config_path: Path) -> "ExperimentArtifacts":
    """Run the research framework from a JSON config.

    Args:
        config_path: Path to a framework config JSON file.

    Returns:
        Written artifact paths.
    """
    try:
        from experiment import ExperimentRunConfig, ResearchExperimentRunner, TabularDataConfig
        from models import MissingMechanismConfig, ObservationModelConfig, SparseGPConfig
        from pipeline import MissingnessTrainingConfig, SilenceAwareIDSConfig, StateTrainingConfig
        from policy import InformationPolicyConfig
        from reliability import ConformalConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Research framework dependencies are not installed. "
            "Install requirements from requirements-research.txt first."
        ) from exc

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent
    data_payload = dict(payload.get("data", {}))
    output_path = (base_dir / payload.get("output_path", "outputs/framework_run/summary.json")).resolve()
    run_payload = dict(payload.get("run", {}))
    pipeline_payload = dict(payload.get("pipeline", {}))

    data_path = (base_dir / data_payload.pop("data_path", "sample_weather.csv")).resolve()
    data_config = TabularDataConfig(data_path=data_path, **data_payload)
    pipeline_config = SilenceAwareIDSConfig(
        state=SparseGPConfig(**pipeline_payload.pop("state")),
        observation=ObservationModelConfig(**pipeline_payload.pop("observation", {})),
        missingness=MissingMechanismConfig(**pipeline_payload.pop("missingness", {})),
        policy=InformationPolicyConfig(**pipeline_payload.pop("policy", {})),
        reliability=ConformalConfig(**pipeline_payload.pop("reliability", {})),
        state_training=StateTrainingConfig(**pipeline_payload.pop("state_training", {})),
        missingness_training=MissingnessTrainingConfig(**pipeline_payload.pop("missingness_training", {})),
        **pipeline_payload,
    )
    run_config = ExperimentRunConfig(
        variant_names=tuple(run_payload.get("variant_names", ())),
        sensitivity_logit_scales=tuple(run_payload.get("sensitivity_logit_scales", (0.5, 1.0, 1.5, 2.0))),
        policy_budget=run_payload.get("policy_budget"),
        max_selections=run_payload.get("max_selections"),
        use_evaluation_as_candidate_pool=bool(run_payload.get("use_evaluation_as_candidate_pool", True)),
        seed=int(run_payload.get("seed", 0)),
        deterministic_algorithms=bool(run_payload.get("deterministic_algorithms", False)),
        torch_num_threads=run_payload.get("torch_num_threads"),
        matmul_precision=str(run_payload.get("matmul_precision", "high")),
        prediction_batch_size=run_payload.get("prediction_batch_size"),
        benchmark_expansion_factor=int(run_payload.get("benchmark_expansion_factor", 1)),
        benchmark_candidate_pool_size=run_payload.get("benchmark_candidate_pool_size"),
    )

    runner = ResearchExperimentRunner(data_config, pipeline_config, run_config)
    return runner.write_artifacts(output_path)
