from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import ResearchExperimentRunner, _deserialize_fit_summary
from pipeline import SilenceAwareIDS, _slice_metadata
from task_cli.framework import load_framework_config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _emit(stage_path: Path, step: str, **payload: Any) -> None:
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "step": step,
        "payload": payload,
    }
    with stage_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect full framework base metrics from cached prepared state.")
    parser.add_argument("--config", required=True, help="Framework config JSON path.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    data_config, pipeline_config, run_config, output_path = load_framework_config(config_path)
    runner = ResearchExperimentRunner(data_config, pipeline_config, run_config)

    output_dir = output_path.parent
    stage_log = output_dir / "collect_framework_base_metrics_progress.jsonl"
    note_path = output_dir / "framework_base_metrics_collected.json"
    progress_path, checkpoint_path = runner._progress_paths(output_path)
    prepared_cache_path = runner._prepared_cache_path(output_path)

    _emit(stage_log, "start", config=str(config_path))
    checkpoint = runner._load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise RuntimeError(f"No compatible checkpoint found: {checkpoint_path}")
    cached_prepared = runner._try_load_prepared_cache(prepared_cache_path)
    if cached_prepared is None:
        _emit(stage_log, "prepare_data_start")
        prepared_raw = runner.prepare_data()
        prepared, row_caps = runner._maybe_cap_prepared_data(prepared_raw)
        runner._save_prepared_cache(prepared_cache_path, prepared, row_caps)
        _emit(stage_log, "prepare_data_complete", source="fresh")
    else:
        prepared, row_caps = cached_prepared
        _emit(stage_log, "prepare_data_complete", source="cache")

    reproducibility = runner._configure_runtime()
    runtime_environment = runner._runtime_environment()
    resolved_pipeline_config = runner._resolve_pipeline_config(prepared)
    resolved_pipeline_config = runner._stabilize_pipeline_config_for_large_runs(
        resolved_pipeline_config,
        row_caps=row_caps,
    )
    pipeline = SilenceAwareIDS(resolved_pipeline_config)
    if checkpoint.get("pipeline_state") is not None:
        runner._restore_pipeline_checkpoint(pipeline, checkpoint.get("pipeline_state"))
    fit_summary = _deserialize_fit_summary(checkpoint.get("fit_summary"))
    inference_batch_size = runner._effective_inference_batch_size(row_caps)
    _emit(stage_log, "pipeline_restored", batch_size=inference_batch_size)

    observed_mask = prepared.evaluation.y.isfinite()
    observed_index = observed_mask.nonzero(as_tuple=False).squeeze(-1)
    X_eval = prepared.evaluation.X[observed_index]
    y_eval = prepared.evaluation.y[observed_index]
    context_eval = None if prepared.evaluation.context is None else prepared.evaluation.context[observed_index]
    M_eval = None if prepared.evaluation.M is None else prepared.evaluation.M[observed_index]
    S_eval = None if prepared.evaluation.S is None else prepared.evaluation.S[observed_index]
    sensor_metadata_eval = _slice_metadata(prepared.evaluation.sensor_metadata, observed_index)
    _emit(stage_log, "observed_subset_ready", observed_rows=int(X_eval.shape[0]))

    if pipeline.use_m3:
        _emit(stage_log, "predictive_summary_start", path="missingness_aware_state_summary")
        mu, var, p_miss = pipeline.missingness_aware_state_summary(
            X_eval,
            y=y_eval,
            context=context_eval,
            M=M_eval,
            S=S_eval,
            sensor_metadata=sensor_metadata_eval,
            batch_size=inference_batch_size,
            logit_scale=1.0,
            include_observation_noise=(
                pipeline.config.reliability.prediction_target == "observation"
            ),
        )
    else:
        _emit(stage_log, "predictive_summary_start", path="predict_state")
        mu, var = pipeline.predict_state(
            X_eval,
            batch_size=inference_batch_size,
            include_observation_noise=(
                pipeline.config.reliability.prediction_target == "observation"
            ),
        )
        p_miss = None
    _emit(
        stage_log,
        "predictive_summary_complete",
        mean_variance=float(var.clamp_min(1e-6).mean().item()),
        mean_missingness=None if p_miss is None else float(p_miss.mean().item()),
    )

    metrics = pipeline.reliability_model.evaluate_gaussian_predictions(
        mu=mu,
        var=var,
        y_true=y_eval,
    )
    base_metrics: dict[str, Any] = {
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "crps": metrics.crps,
        "log_score": metrics.log_score,
    }
    if p_miss is not None:
        base_metrics["mean_missingness_proba"] = float(p_miss.mean().item())
    _emit(
        stage_log,
        "point_metrics_complete",
        rmse=float(metrics.rmse),
        mae=float(metrics.mae),
        crps=float(metrics.crps),
        log_score=float(metrics.log_score),
    )
    _write_json(
        note_path,
        {
            "config_path": str(config_path),
            "stage": "point_metrics_complete",
            "base_metrics": base_metrics,
            "inference_batch_size": inference_batch_size,
        },
    )

    if pipeline.use_m5 and pipeline.reliability_model is not None and pipeline.reliability_model.is_calibrated:
        silence_features = None
        if pipeline.config.use_m2 and (
            S_eval is None or pipeline.config.reliability.mode in {"relational_adaptive", "graph_corel"}
        ):
            _emit(stage_log, "silence_detection_start")
            silence_features = pipeline.detect_silence(
                X_eval,
                y_eval,
                context=context_eval,
                sensor_metadata=sensor_metadata_eval,
                batch_size=inference_batch_size,
            )
            if S_eval is None:
                S_eval = silence_features["dynamic_silence"].float()
            _emit(
                stage_log,
                "silence_detection_complete",
                flagged=int(silence_features["dynamic_silence"].sum().item()),
            )

        lower = None
        upper = None
        interval_metadata: dict[str, Any] | None = None
        mode = pipeline.config.reliability.mode
        if mode == "adaptive":
            pipeline.reliability_model.reset_adaptation()
            _emit(stage_log, "intervals_start", mode=mode)
            lower, upper, interval_metadata = pipeline.reliability_model.predict_interval_adaptive(
                mu_test=mu,
                var_test=var,
                y_true=y_eval,
            )
        elif mode in {"relational_adaptive", "graph_corel"}:
            pipeline.reliability_model.reset_adaptation()
            _emit(stage_log, "relational_features_start", mode=mode)
            node_features = pipeline._relational_node_features(  # type: ignore[attr-defined]
                X_eval,
                y=y_eval,
                context=context_eval,
                sensor_metadata=sensor_metadata_eval,
                silence_features=silence_features,
                batch_size=inference_batch_size,
            )
            _emit(stage_log, "relational_features_complete", feature_dim=int(node_features.shape[-1]))
            _emit(stage_log, "intervals_start", mode=mode)
            lower, upper, interval_metadata = pipeline.reliability_model.predict_interval_adaptive(
                mu_test=mu,
                var_test=var,
                y_true=y_eval,
                node_features=node_features,
            )
        else:
            _emit(stage_log, "intervals_start", mode="split")
            lower, upper, interval_metadata = pipeline.reliability_model.predict_interval(
                mu_test=mu,
                var_test=var,
            )
        _emit(
            stage_log,
            "intervals_complete",
            covered_rows=0 if lower is None else int(lower.shape[0]),
            interval_width=None if lower is None or upper is None else float((upper - lower).mean().item()),
        )

        metrics_with_intervals = pipeline.reliability_model.evaluate_gaussian_predictions(
            mu=mu,
            var=var,
            y_true=y_eval,
            lower=lower,
            upper=upper,
        )
        base_metrics["coverage"] = metrics_with_intervals.coverage
        base_metrics["interval_width"] = metrics_with_intervals.interval_width
        if interval_metadata is not None and "final_epsilon" in interval_metadata:
            base_metrics["final_adaptive_epsilon"] = float(interval_metadata["final_epsilon"])
        if interval_metadata is not None and "mean_neighbor_error" in interval_metadata:
            base_metrics["mean_neighbor_error"] = float(interval_metadata["mean_neighbor_error"])
        if interval_metadata is not None and "mean_graph_quantile" in interval_metadata:
            base_metrics["mean_graph_quantile"] = float(interval_metadata["mean_graph_quantile"])

    dataset_summary = {
        "train_rows": int(prepared.train.X.shape[0]),
        "calibration_rows": int(prepared.calibration.X.shape[0]),
        "evaluation_rows": int(prepared.evaluation.X.shape[0]),
        "row_caps": row_caps,
        "effective_reliability_mode": resolved_pipeline_config.reliability.mode,
        "input_dim": int(prepared.full.X.shape[1]),
        "context_dim": 0 if prepared.full.context is None else int(prepared.full.context.shape[1]),
        "metadata_cardinalities": prepared.metadata_cardinalities,
        "feature_schema": prepared.feature_schema,
    }
    runner._write_progress_payload(
        path=progress_path,
        summary_path=output_path,
        status="running",
        stage="base_metrics_complete",
        runtime_environment=runtime_environment,
        reproducibility=reproducibility,
        dataset_summary=dataset_summary,
        fit_summary=fit_summary,
        base_metrics=base_metrics,
    )
    runner._save_checkpoint(
        checkpoint_path,
        stage="base_metrics_complete",
        pipeline=pipeline,
        include_pipeline_state=False,
        fit_summary=fit_summary,
        base_metrics=base_metrics,
        dataset_summary=dataset_summary,
        runtime_environment=runtime_environment,
        reproducibility=reproducibility,
    )
    _write_json(
        note_path,
        {
            "config_path": str(config_path),
            "stage": "completed",
            "base_metrics": base_metrics,
            "inference_batch_size": inference_batch_size,
        },
    )
    _emit(stage_log, "completed", metric_keys=sorted(base_metrics.keys()))
    print(json.dumps(base_metrics, indent=2, ensure_ascii=False))
    print(f"Wrote: {note_path}")


if __name__ == "__main__":
    main()
