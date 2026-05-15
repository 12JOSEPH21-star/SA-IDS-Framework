from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_station_registry import build_station_registry


class StationRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_csv = self.root / "joint_weather_network_q1.csv"
        self.output_csv = self.root / "station_registry_q1.csv"
        self.summary_json = self.root / "station_registry_summary.json"
        with self.input_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "station_id",
                    "source",
                    "latitude",
                    "longitude",
                    "elevation",
                    "temperature",
                    "qc_hour_bucket_availability_ratio",
                    "era5_temperature",
                    "era5_land_sea_mask",
                    "event_station_active",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "timestamp": "2025-01-01T00:00",
                        "station_id": "aws_1",
                        "source": "aws",
                        "latitude": "38.1",
                        "longitude": "128.8",
                        "elevation": "25.0",
                        "temperature": "4.0",
                        "qc_hour_bucket_availability_ratio": "0.5",
                        "era5_temperature": "3.8",
                        "era5_land_sea_mask": "0.4",
                        "event_station_active": "1",
                    },
                    {
                        "timestamp": "2025-01-01T01:00",
                        "station_id": "aws_1",
                        "source": "aws",
                        "latitude": "38.1",
                        "longitude": "128.8",
                        "elevation": "25.0",
                        "temperature": "",
                        "qc_hour_bucket_availability_ratio": "1.0",
                        "era5_temperature": "",
                        "era5_land_sea_mask": "0.4",
                        "event_station_active": "0",
                    },
                    {
                        "timestamp": "2025-01-01T00:00",
                        "station_id": "asos_2",
                        "source": "asos",
                        "latitude": "37.7",
                        "longitude": "127.1",
                        "elevation": "600.0",
                        "temperature": "1.0",
                        "qc_hour_bucket_availability_ratio": "1.0",
                        "era5_temperature": "",
                        "era5_land_sea_mask": "1.0",
                        "event_station_active": "",
                    },
                ]
            )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_build_station_registry_writes_summary_outputs(self) -> None:
        build_station_registry(self.input_csv, self.output_csv, self.summary_json)
        self.assertTrue(self.output_csv.exists())
        self.assertTrue(self.summary_json.exists())

        with self.output_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        first = rows[0]
        self.assertIn(first["region_bin"], {"northeast", "northwest", "southeast", "southwest"})
        self.assertIn(first["climate_zone"], {"coastal", "highland", "inland", "international_synoptic"})

        summary = json.loads(self.summary_json.read_text(encoding="utf-8"))
        self.assertEqual(summary["station_count"], 2)
        self.assertIn("coastal", summary["climate_zone_counts"])
        self.assertIn("highland", summary["climate_zone_counts"])


if __name__ == "__main__":
    unittest.main()
