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
from task_cli.kma_event import (
    WARNING_API_PAGE,
    API_HUB_GUIDE,
    KmaEventDownloadConfig,
    download_kma_events,
)


class KmaEventCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_download_kma_warning_history_is_standardized(self) -> None:
        output_dir = self.root / "event"

        def fetcher(url: str, timeout: int) -> bytes:
            del url, timeout
            text = "\n".join(
                [
                    "# TM_FC TM_EF TM_ST TM_ED REG_ID REG_KO STN WRN LVL CMD TM_SEQ",
                    "202501010600 202501010700 202501010700 202501011200 11 서울 108 R 1 I 7",
                ]
            )
            return text.encode("utf-8")

        artifacts = download_kma_events(
            KmaEventDownloadConfig(
                source="warning_history",
                output_dir=output_dir,
                start_datetime=datetime(2025, 1, 1, 0, 0),
                end_datetime=datetime(2025, 1, 2, 0, 0),
                service_key="example-auth-key",
            ),
            fetcher=fetcher,
        )

        self.assertEqual(artifacts.row_count, 1)
        self.assertTrue(artifacts.standardized_csv_path is not None and artifacts.standardized_csv_path.exists())
        with artifacts.standardized_csv_path.open("r", encoding="utf-8", newline="") as handle:  # type: ignore[union-attr]
            row = next(csv.DictReader(handle))
        self.assertEqual(row["source"], "warning_history")
        self.assertEqual(row["warning_code"], "R")
        self.assertEqual(row["warning_level"], "1")
        self.assertEqual(row["region_short_name"], "서울")
        manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
        self.assertIn(WARNING_API_PAGE, manifest["official_sources"])
        self.assertIn(API_HUB_GUIDE, manifest["official_sources"])

    def test_app_supports_kma_event_download_dry_run(self) -> None:
        output = io.StringIO()
        manifest_dir = self.root / "event_dry_run"
        with redirect_stdout(output):
            code = main(
                [
                    "kma-event-download",
                    "--source",
                    "warning_history",
                    "--output-dir",
                    str(manifest_dir),
                    "--start-datetime",
                    "2025-01-01T00:00",
                    "--end-datetime",
                    "2025-03-31T23:00",
                    "--warning-codes",
                    "R",
                    "W",
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)
        manifest = json.loads((manifest_dir / "kma_event_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"], "warning_history")
        self.assertEqual(manifest["warning_codes"], ["R", "W"])
        self.assertIn("Planned KMA event warning_history download", output.getvalue())


if __name__ == "__main__":
    unittest.main()
