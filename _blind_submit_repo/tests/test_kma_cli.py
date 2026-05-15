from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path

from task_cli.app import main
from task_cli.kma import (
    ASOS_API_HUB_PAGE,
    ASOS_OFFICIAL_PAGE,
    AWS_API_HUB_GUIDE,
    AWS_API_HUB_PAGE,
    KmaDownloadConfig,
    download_kma,
    write_station_metadata_template,
)


class KmaCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_write_station_metadata_template_writes_defaults(self) -> None:
        template_path = self.root / "stations.csv"

        written = write_station_metadata_template(
            template_path,
            station_ids=("108", "159"),
            source="asos_hourly",
        )

        self.assertEqual(written, template_path)
        with template_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["station_id"], "108")
        self.assertEqual(rows[0]["sensor_type"], "asos")
        self.assertEqual(rows[0]["sensor_group"], "surface")

    def test_download_kma_asos_writes_raw_standardized_and_framework_config(self) -> None:
        output_dir = self.root / "kma"
        metadata_path = self.root / "station_metadata.csv"
        metadata_path.write_text(
            "\n".join(
                [
                    "station_id,latitude,longitude,elevation,cost,sensor_type,sensor_group,sensor_modality,site_type,maintenance_state,maintenance_age",
                    "108,37.571,126.966,86.4,1.5,asos,surface,synoptic_surface,urban,nominal,2.0",
                ]
            ),
            encoding="utf-8",
        )

        requested_urls: list[str] = []

        def fetcher(url: str, timeout: int) -> bytes:
            del timeout
            requested_urls.append(url)
            payload = {
                "response": {
                    "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
                    "body": {
                        "totalCount": 1,
                        "items": {
                            "item": [
                                {
                                    "tm": "2025-01-01 00:00",
                                    "stnId": "108",
                                    "ta": "1.2",
                                    "hm": "55.0",
                                    "ps": "1012.3",
                                    "ws": "2.4",
                                    "wd": "180",
                                    "rn": "0.0",
                                }
                            ]
                        },
                    },
                }
            }
            return json.dumps(payload).encode("utf-8")

        artifacts = download_kma(
            KmaDownloadConfig(
                source="asos_hourly",
                output_dir=output_dir,
                standardized_csv_path=output_dir / "asos_framework.csv",
                framework_config_path=self.root / "framework_kma.json",
                metadata_csv_path=metadata_path,
                service_key="encoded%2Fkey",
                start_date=date(2025, 1, 1),
                end_date=date(2025, 1, 1),
                station_ids=("108",),
            ),
            fetcher=fetcher,
        )

        self.assertEqual(artifacts.row_count, 1)
        self.assertTrue(artifacts.raw_csv_path is not None and artifacts.raw_csv_path.exists())
        self.assertTrue(
            artifacts.standardized_csv_path is not None and artifacts.standardized_csv_path.exists()
        )
        self.assertTrue(
            artifacts.framework_config_path is not None and artifacts.framework_config_path.exists()
        )
        self.assertTrue(requested_urls)
        self.assertIn("stnIds=108", requested_urls[0])
        self.assertIn("startDt=20250101", requested_urls[0])

        with artifacts.standardized_csv_path.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["timestamp"], "2025-01-01T00:00")
        self.assertEqual(row["station_id"], "108")
        self.assertEqual(row["latitude"], "37.571")
        self.assertEqual(row["longitude"], "126.966")
        self.assertEqual(row["temperature"], "1.2")
        self.assertEqual(row["sensor_type"], "asos")

        framework_payload = json.loads(artifacts.framework_config_path.read_text(encoding="utf-8"))
        self.assertEqual(framework_payload["preset"], "aws_network")
        self.assertEqual(framework_payload["data"]["target_column"], "temperature")

        manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["auth_source"], "cli")
        self.assertIn(ASOS_OFFICIAL_PAGE, manifest["official_sources"])

    def test_download_kma_asos_apihub_text_response_is_standardized(self) -> None:
        output_dir = self.root / "kma_asos_apihub"

        requested_urls: list[str] = []

        def fetcher(url: str, timeout: int) -> bytes:
            del timeout
            requested_urls.append(url)
            if "stn_inf.php" in url:
                text = "\n".join(
                    [
                        "#START7777",
                        "108 126.9667 37.5714 101 86.4 SEOUL",
                    ]
                )
                return text.encode("utf-8")
            text = "\n".join(
                [
                    "# YYMMDDHHMI STN WD WS GST_WD GST_WS GST_TM PA PS PT PR TA TD HM PV RN RN_DAY RN_JUN RN_INT SD_HR3 SD_DAY SD_TOT WC WP WW CA_TOT CA_MID CH_MIN CT CT_TOP CT_MID CT_LOW VS SS SI ST_GD TS TE_005 TE_01 TE_02 TE_03 ST_SEA WH BF IR IX",
                    "202501010000 108 200 2.4 0 0.0 0 1012.1 1021.0 0.0 0.0 1.2 -5.0 55 4.2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0 0 0 3 0 1200 SC 3000 0 0 20000 0.0 0.0 0.0 0.1 0.2 0.3 0.4 10.0 0.0 2 0.0 0",
                ]
            )
            return text.encode("utf-8")

        artifacts = download_kma(
            KmaDownloadConfig(
                source="asos_hourly_apihub",
                output_dir=output_dir,
                standardized_csv_path=output_dir / "asos_apihub_framework.csv",
                framework_config_path=self.root / "framework_kma_apihub.json",
                service_key="example-auth-key",
                start_date=date(2025, 1, 1),
                end_date=date(2025, 1, 1),
                station_ids=("108",),
            ),
            fetcher=fetcher,
        )

        self.assertEqual(artifacts.row_count, 1)
        self.assertTrue(requested_urls)
        self.assertIn("authKey=example-auth-key", requested_urls[0])
        self.assertIn("tm1=202501010000", requested_urls[0])
        self.assertIn("tm2=202501012300", requested_urls[0])
        self.assertIn("stn=108", requested_urls[0])
        self.assertTrue(
            artifacts.resolved_metadata_csv_path is not None and artifacts.resolved_metadata_csv_path.exists()
        )
        with artifacts.standardized_csv_path.open("r", encoding="utf-8", newline="") as handle:  # type: ignore[union-attr]
            row = next(csv.DictReader(handle))
        self.assertEqual(row["timestamp"], "2025-01-01T00:00")
        self.assertEqual(row["station_id"], "108")
        self.assertEqual(row["temperature"], "1.2")
        self.assertEqual(row["humidity"], "55")
        self.assertEqual(row["pressure"], "1021.0")
        self.assertEqual(row["wind_speed"], "2.4")
        self.assertEqual(row["wind_direction"], "200")
        self.assertEqual(row["latitude"], "37.5714")
        self.assertEqual(row["longitude"], "126.9667")
        self.assertEqual(row["sensor_type"], "asos")
        manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
        self.assertIn(ASOS_API_HUB_PAGE, manifest["official_sources"])
        self.assertIn(AWS_API_HUB_GUIDE, manifest["official_sources"])

    def test_download_kma_aws_dry_run_writes_manifest_and_template(self) -> None:
        output_dir = self.root / "kma"
        template_path = self.root / "metadata_template.csv"

        artifacts = download_kma(
            KmaDownloadConfig(
                source="aws_recent_1min",
                output_dir=output_dir,
                metadata_template_path=template_path,
                station_ids=("400", "401"),
                aws_datetime=datetime(2025, 1, 2, 9, 0),
                dry_run=True,
            )
        )

        self.assertTrue(artifacts.manifest_path.exists())
        self.assertTrue(template_path.exists())
        self.assertTrue(artifacts.raw_csv_path is None)
        manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["auth_source"], "missing")
        self.assertEqual(len(manifest["request_status"]), 2)
        self.assertIn("stn=400", manifest["request_status"][0]["url"])
        self.assertIn("stn=401", manifest["request_status"][1]["url"])
        self.assertIn(AWS_API_HUB_PAGE, manifest["official_sources"])
        self.assertIn(AWS_API_HUB_GUIDE, manifest["official_sources"])

    def test_download_kma_aws_apihub_text_response_is_standardized(self) -> None:
        output_dir = self.root / "kma"
        metadata_path = self.root / "station_metadata.csv"
        metadata_path.write_text(
            "\n".join(
                [
                    "station_id,latitude,longitude,elevation,cost,sensor_type,sensor_group,sensor_modality,site_type,maintenance_state,maintenance_age",
                    "400,35.170,128.572,12.0,0.8,aws,surface,automatic_weather_station,coastal,nominal,1.0",
                ]
            ),
            encoding="utf-8",
        )

        requested_urls: list[str] = []

        def fetcher(url: str, timeout: int) -> bytes:
            del timeout
            requested_urls.append(url)
            text = "\n".join(
                [
                    "# START7777",
                    "TM,STN,TA,HM,PA,WS,WD,RN",
                    "202501020900,400,3.2,45.0,1013.5,5.1,270,0.0",
                    "# END7777",
                ]
            )
            return text.encode("utf-8")

        artifacts = download_kma(
            KmaDownloadConfig(
                source="aws_recent_1min",
                output_dir=output_dir,
                standardized_csv_path=output_dir / "aws_framework.csv",
                metadata_csv_path=metadata_path,
                service_key="example-auth-key",
                station_ids=("400",),
                aws_datetime=datetime(2025, 1, 2, 9, 0),
            ),
            fetcher=fetcher,
        )

        self.assertEqual(artifacts.row_count, 1)
        self.assertTrue(requested_urls)
        self.assertIn("authKey=example-auth-key", requested_urls[0])
        self.assertIn("tm2=202501020900", requested_urls[0])
        self.assertIn("stn=400", requested_urls[0])

        with artifacts.standardized_csv_path.open("r", encoding="utf-8", newline="") as handle:  # type: ignore[union-attr]
            row = next(csv.DictReader(handle))
        self.assertEqual(row["timestamp"], "2025-01-02T09:00")
        self.assertEqual(row["station_id"], "400")
        self.assertEqual(row["temperature"], "3.2")
        self.assertEqual(row["humidity"], "45.0")
        self.assertEqual(row["pressure"], "1013.5")
        self.assertEqual(row["wind_speed"], "5.1")
        self.assertEqual(row["wind_direction"], "270")
        self.assertEqual(row["latitude"], "35.170")
        self.assertEqual(row["sensor_type"], "aws")

    def test_download_kma_aws_apihub_uses_wd10_ws10_fields(self) -> None:
        output_dir = self.root / "kma"

        def fetcher(url: str, timeout: int) -> bytes:
            del url, timeout
            text = "\n".join(
                [
                    "# START7777",
                    "202501020900,401,10.0,1.0,20.0,2.0,280.0,7.5,1.0,0.0,0.0,0.0,0.0,0.0,65.0,1010.0,1014.0,-3.0,=",
                ]
            )
            return text.encode("utf-8")

        artifacts = download_kma(
            KmaDownloadConfig(
                source="aws_recent_1min",
                output_dir=output_dir,
                standardized_csv_path=output_dir / "aws_framework.csv",
                service_key="example-auth-key",
                station_ids=("401",),
                aws_datetime=datetime(2025, 1, 2, 9, 0),
            ),
            fetcher=fetcher,
        )

        with artifacts.standardized_csv_path.open("r", encoding="utf-8", newline="") as handle:  # type: ignore[union-attr]
            row = next(csv.DictReader(handle))
        self.assertEqual(row["wind_direction"], "280.0")
        self.assertEqual(row["wind_speed"], "7.5")

    def test_download_kma_aws_auto_station_metadata_populates_coordinates(self) -> None:
        output_dir = self.root / "kma"

        def fetcher(url: str, timeout: int) -> bytes:
            del timeout
            if "stn_inf.php" in url:
                text = "\n".join(
                    [
                        "#START7777",
                        " 400  128.57282000   35.17019000 41211110     34.97     10.00 4155 155 TEST ---- 11H20301 4812510100 171",
                    ]
                )
                return text.encode("utf-8")
            text = "\n".join(
                [
                    "# START7777",
                    "202501020900,400,10.0,1.0,20.0,2.0,280.0,7.5,1.0,0.0,0.0,0.0,0.0,0.0,65.0,1010.0,1014.0,-3.0,=",
                ]
            )
            return text.encode("utf-8")

        artifacts = download_kma(
            KmaDownloadConfig(
                source="aws_recent_1min",
                output_dir=output_dir,
                standardized_csv_path=output_dir / "aws_framework.csv",
                service_key="example-auth-key",
                station_ids=("400",),
                aws_datetime=datetime(2025, 1, 2, 9, 0),
            ),
            fetcher=fetcher,
        )

        self.assertTrue(
            artifacts.resolved_metadata_csv_path is not None and artifacts.resolved_metadata_csv_path.exists()
        )
        with artifacts.standardized_csv_path.open("r", encoding="utf-8", newline="") as handle:  # type: ignore[union-attr]
            row = next(csv.DictReader(handle))
        self.assertEqual(row["station_id"], "400")
        self.assertEqual(row["latitude"], "35.17019000")
        self.assertEqual(row["longitude"], "128.57282000")
        self.assertEqual(row["elevation"], "34.97")
        manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            Path(manifest["resolved_metadata_csv_path"]).name,
            "aws_station_metadata_auto.csv",
        )

    def test_app_kma_download_command_supports_dry_run(self) -> None:
        output_dir = self.root / "app_run"
        output = io.StringIO()

        with redirect_stdout(output):
            code = main(
                [
                    "kma-download",
                    "--source",
                    "asos_hourly",
                    "--output-dir",
                    str(output_dir),
                    "--station-ids",
                    "108",
                    "--start-date",
                    "2025-01-01",
                    "--end-date",
                    "2025-01-01",
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)
        self.assertTrue((output_dir / "kma_download_manifest.json").exists())

    def test_app_kma_download_command_supports_asos_apihub_dry_run(self) -> None:
        output_dir = self.root / "app_run_apihub"
        output = io.StringIO()

        with redirect_stdout(output):
            code = main(
                [
                    "kma-download",
                    "--source",
                    "asos_hourly_apihub",
                    "--output-dir",
                    str(output_dir),
                    "--station-ids",
                    "108",
                    "--start-date",
                    "2025-01-01",
                    "--end-date",
                    "2025-01-02",
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)
        manifest = json.loads((output_dir / "kma_download_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"], "asos_hourly_apihub")
        self.assertIn(ASOS_API_HUB_PAGE, manifest["official_sources"])
        self.assertIn("Planned KMA asos_hourly_apihub download", output.getvalue())


if __name__ == "__main__":
    unittest.main()
