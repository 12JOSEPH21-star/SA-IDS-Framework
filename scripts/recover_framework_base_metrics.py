from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _recover_base_metrics(events: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    completed_index = None
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        if event.get("stage") == "evaluating_base_pipeline" and event.get("step") == "completed":
            completed_index = index
            break
    if completed_index is None:
        raise RuntimeError("No completed base-evaluation heartbeat event found.")

    recovered: dict[str, Any] = {
        "rmse": None,
        "mae": None,
        "crps": None,
        "log_score": None,
        "coverage": None,
        "interval_width": None,
        "mean_missingness_proba": None,
        "final_adaptive_epsilon": None,
        "mean_graph_quantile": None,
        "mean_neighbor_error": None,
    }
    completed_payload = events[completed_index].get("payload", {})
    metric_keys = list(completed_payload.get("metric_keys", []))

    for index in range(completed_index, -1, -1):
        event = events[index]
        if event.get("stage") != "evaluating_base_pipeline":
            continue
        step = event.get("step")
        payload = event.get("payload", {})
        if step == "metrics_complete":
            for key in ("rmse", "crps", "coverage"):
                if key in payload:
                    recovered[key] = payload[key]
        elif step == "intervals_complete" and "interval_width" in payload:
            recovered["interval_width"] = payload["interval_width"]
        elif step == "predictive_summary_complete" and "mean_missingness" in payload:
            recovered["mean_missingness_proba"] = payload["mean_missingness"]
        elif step in {"evaluation_silence_start", "dispatch", "start"}:
            break

    metadata = {
        "recovered_at": datetime.now().isoformat(timespec="seconds"),
        "recovered_from_heartbeat_step": events[completed_index].get("step"),
        "recovered_metric_keys": metric_keys,
        "missing_metric_keys": [
            key for key in metric_keys if key in recovered and recovered.get(key) is None
        ],
        "latest_completed_timestamp": events[completed_index].get("timestamp"),
    }
    return recovered, metadata


def _update_json_payload(path: Path, base_metrics: dict[str, Any], metadata: dict[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "running"
    payload["stage"] = "base_metrics_complete"
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["error"] = None
    payload["base_metrics"] = base_metrics
    payload["base_metrics_recovered_from_heartbeat"] = metadata
    payload["is_partial"] = True
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _update_checkpoint(path: Path, base_metrics: dict[str, Any], metadata: dict[str, Any]) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["stage"] = "base_metrics_complete"
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["base_metrics"] = base_metrics
    payload["base_metrics_recovered_from_heartbeat"] = metadata
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover framework base metrics from heartbeat traces.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Framework run output directory containing summary/progress/checkpoint/heartbeat files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    heartbeat_path = output_dir / "framework_heartbeat.jsonl"
    progress_path = output_dir / "framework_progress.json"
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "framework_run_checkpoint.pt"
    note_path = output_dir / "framework_base_metrics_recovery.json"

    for path in (heartbeat_path, progress_path, summary_path, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    events = _load_jsonl(heartbeat_path)
    base_metrics, metadata = _recover_base_metrics(events)
    _update_json_payload(progress_path, base_metrics, metadata)
    _update_json_payload(summary_path, base_metrics, metadata)
    _update_checkpoint(checkpoint_path, base_metrics, metadata)
    note_path.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "base_metrics": base_metrics,
                "metadata": metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Recovered base metrics into: {summary_path}")
    print(f"Checkpoint updated: {checkpoint_path}")
    print(f"Recovery note: {note_path}")


if __name__ == "__main__":
    main()
