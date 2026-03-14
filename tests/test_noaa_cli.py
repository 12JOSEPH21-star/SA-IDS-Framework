from __future__ import annotations

import csv
import gzip
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from unittest.mock import patch

from task_cli.app import main
from task_cli.noaa import (
    ISD_HISTORY_URL,
    ISD_LITE_OFFICIAL_PAGE,
    NoaaDownloadConfig,
    NoaaDownloadArtifacts,
    download_noaa_isd,
)


class NoaaCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_download_noaa_isd_writes_standardized_csv_and_config(self) -> None:
        output_dir = self.root / "noaa"

        metadata_csv = "\n".join(
            [
                '"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"',
                '"471080","99999","SEOUL STATION","KS","","RKSS","+37.571","+126.965","86.0","19500101","20251231"',
            ]
        ).encode("utf-8")
        station_text = "\n".join(
            [
                "2025 01 01 00     0   -53 10221    45    17     6 -9999 -9999",
                "2025 01 01 01    12   -44 10225    51    16     7 -9999 -9999",
            ]
        )
        station_payload = gzip.compress(station_text.encode("utf-8"))

        def fetcher(url: str, timeout: int) -> bytes:
            del timeout
            if url == ISD_HISTORY_URL:
                return metadata_csv
            if url.endswith("/2025/471080-99999-2025.gz"):
                return station_payload
            raise AssertionError(f"unexpected url: {url}")

        artifacts = download_noaa_isd(
            NoaaDownloadConfig(
                start_date=date(2025, 1, 1),
                end_date=date(2025, 1, 1),
                output_dir=output_dir,
                station_ids=("471080-99999",),
                framework_config_path=self.root / "framework_isd.json",
            ),
            fetcher=fetcher,
        )

        self.assertEqual(artifacts.station_count, 1)
        self.assertEqual(artifacts.row_count, 2)
        self.assertTrue(artifacts.raw_csv_path is not None and artifacts.raw_csv_path.exists())
        self.assertTrue(
            artifacts.standardized_csv_path is not None and artifacts.standardized_csv_path.exists()
        )
        self.assertTrue(
            artifacts.framework_config_path is not None and artifacts.framework_config_path.exists()
        )

        with artifacts.standardized_csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["station_id"], "471080-99999")
        self.assertEqual(rows[0]["air_temperature"], "0.0")
        self.assertEqual(rows[0]["dew_point_temperature"], "-5.3")
        self.assertEqual(rows[0]["sea_level_pressure"], "1022.1")
        self.assertEqual(rows[0]["wind_speed"], "1.7")
        self.assertEqual(rows[0]["site_type"], "airport")
        self.assertEqual(rows[0]["sensor_type"], "isd_station")

        manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["station_count"], 1)
        self.assertEqual(manifest["row_count"], 2)
        self.assertIn(ISD_LITE_OFFICIAL_PAGE, manifest["official_sources"])

    def test_noaa_download_cli_dry_run_writes_manifest(self) -> None:
        output_dir = self.root / "noaa_cli"
        output = io.StringIO()
        manifest_path = output_dir / "noaa_download_manifest.json"
        metadata_path = output_dir / "isd_station_metadata.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"dry_run": True}), encoding="utf-8")
        metadata_path.write_text("station_id\n", encoding="utf-8")
        artifacts = NoaaDownloadArtifacts(
            output_dir=output_dir,
            manifest_path=manifest_path,
            metadata_csv_path=metadata_path,
            raw_csv_path=None,
            standardized_csv_path=None,
            framework_config_path=self.root / "framework_isd.json",
            station_count=2,
            row_count=0,
            dry_run=True,
        )
        with patch("task_cli.noaa.download_noaa_isd", return_value=artifacts), redirect_stdout(output):
            code = main(
                [
                    "noaa-download",
                    "--start-date",
                    "2025-01-01",
                    "--end-date",
                    "2025-01-31",
                    "--output-dir",
                    str(output_dir),
                    "--country-code",
                    "KS",
                    "--max-stations",
                    "2",
                    "--framework-config",
                    str(self.root / "framework_isd.json"),
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)
        self.assertIn("Planned NOAA ISD download", output.getvalue())
        self.assertTrue((output_dir / "isd_station_metadata.csv").exists())


if __name__ == "__main__":
    unittest.main()
