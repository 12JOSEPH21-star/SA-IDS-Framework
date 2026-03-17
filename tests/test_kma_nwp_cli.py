from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from task_cli.app import main
from task_cli.kma_nwp import (
    API_HUB_GUIDE,
    NWP_API_PAGE,
    KmaNwpDownloadConfig,
    download_kma_nwp,
)


class KmaNwpCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_download_kma_nwp_summarizes_json_payload(self) -> None:
        output_dir = self.root / "nwp"

        def fetcher(url: str, timeout: int) -> bytes:
            del url, timeout
            payload = {
                "response": {
                    "body": {
                        "items": {
                            "item": [
                                {
                                    "baseTime": "202603150000",
                                    "fcstTime": "202603150600",
                                    "dataTypeCd": "Temp",
                                    "gridKm": "1.5",
                                    "xdim": "2",
                                    "ydim": "2",
                                    "x0": "0",
                                    "y0": "0",
                                    "unit": "C",
                                    "value": "1.0,2.0,3.0,4.0",
                                }
                            ]
                        }
                    }
                }
            }
            return json.dumps(payload).encode("utf-8")

        artifacts = download_kma_nwp(
            KmaNwpDownloadConfig(
                source="ldaps_unis_all",
                output_dir=output_dir,
                start_base_time=datetime(2026, 3, 15, 0, 0),
                end_base_time=datetime(2026, 3, 15, 0, 0),
                service_key="example-auth-key",
            ),
            fetcher=fetcher,
        )

        self.assertEqual(artifacts.row_count, 3)
        self.assertTrue(artifacts.summary_csv_path is not None and artifacts.summary_csv_path.exists())
        with artifacts.summary_csv_path.open("r", encoding="utf-8", newline="") as handle:  # type: ignore[union-attr]
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["data_type_code"], "Temp")
        self.assertEqual(rows[0]["value_count"], "4")
        manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
        self.assertIn(NWP_API_PAGE, manifest["official_sources"])
        self.assertIn(API_HUB_GUIDE, manifest["official_sources"])

    def test_app_supports_kma_nwp_download_dry_run(self) -> None:
        output = io.StringIO()
        manifest_dir = self.root / "nwp_dry_run"
        with redirect_stdout(output):
            code = main(
                [
                    "kma-nwp-download",
                    "--source",
                    "rdaps_unis_all",
                    "--output-dir",
                    str(manifest_dir),
                    "--start-base-time",
                    "2026-03-15T00:00",
                    "--end-base-time",
                    "2026-03-15T12:00",
                    "--lead-hours",
                    "0",
                    "6",
                    "--data-type-code",
                    "Temp",
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)
        manifest = json.loads((manifest_dir / "kma_nwp_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"], "rdaps_unis_all")
        self.assertEqual(manifest["data_type_code"], "Temp")
        self.assertIn("Planned KMA NWP rdaps_unis_all download", output.getvalue())


if __name__ == "__main__":
    unittest.main()
