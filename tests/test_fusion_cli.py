from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from task_cli.app import main
from task_cli.fusion import JointBuildConfig, build_joint_dataset


class JointFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_build_joint_dataset_merges_sources_and_enriches_era5(self) -> None:
        aws_csv = self.root / "aws.csv"
        aws_csv.write_text(
            "\n".join(
                [
                    "timestamp,station_id,latitude,longitude,temperature,humidity,pressure,wind_speed,wind_direction,precipitation,elevation,cost,sensor_type,sensor_group,sensor_modality,site_type,maintenance_state,maintenance_age,source,raw_timestamp",
                    "2025-01-01T00:00,90,38.25,128.56,0.1,50.0,1019.0,1.5,255.8,0.0,17.5,,aws,surface,automatic_weather_station,coastal,nominal,1.0,aws_recent_1min,202501010000",
                ]
            ),
            encoding="utf-8",
        )
        noaa_csv = self.root / "noaa.csv"
        noaa_csv.write_text(
            "\n".join(
                [
                    "timestamp,station_id,latitude,longitude,air_temperature,dew_point_temperature,sea_level_pressure,wind_speed,wind_direction,precipitation_1hr,elevation,station_age_years,sensor_cost,sensor_type,sensor_group,sensor_modality,site_type,maintenance_state,source,raw_station_name,icao",
                    "2025-01-01T01:00,471080-99999,37.57,126.97,-1.0,-3.0,1020.0,2.0,180,0.0,86.0,20.0,1.0,isd_station,surface,synoptic_surface,airport,,noaa_isd_lite,SEOUL STATION,RKSS",
                ]
            ),
            encoding="utf-8",
        )
        era5_csv = self.root / "era5.csv"
        era5_csv.write_text(
            "\n".join(
                [
                    "time,grid_id,latitude,longitude,t2m,u10,v10,sp,tp,orography,land_sea_mask,source_file",
                    "2025-01-01 00:00:00,era5_a,38.250,128.500,273.25,1.0,2.0,101500,0.001,10.0,0.2,file.nc",
                    "2025-01-01 01:00:00,era5_b,37.500,127.000,274.25,1.5,2.5,101600,0.002,20.0,0.3,file.nc",
                ]
            ),
            encoding="utf-8",
        )

        artifacts = build_joint_dataset(
            JointBuildConfig(
                aws_csv_path=aws_csv,
                noaa_csv_path=noaa_csv,
                era5_csv_path=era5_csv,
                output_dir=self.root / "joint",
                framework_config_path=self.root / "joint" / "framework_joint.json",
            )
        )

        self.assertEqual(artifacts.row_count, 2)
        with artifacts.output_csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["station_id"], "aws_90")
        self.assertEqual(rows[0]["era5_temperature"], "0.10")
        self.assertEqual(rows[0]["dew_point_temperature"], "-9.1")
        self.assertEqual(rows[1]["station_id"], "noaa_471080-99999")
        self.assertEqual(rows[1]["humidity"], "86.3")
        self.assertEqual(rows[1]["era5_pressure"], "1016.00")
        self.assertTrue(artifacts.framework_config_path is not None and artifacts.framework_config_path.exists())

        payload = json.loads(artifacts.framework_config_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        self.assertEqual(payload["preset"], "joint_weather_network")
        self.assertIn("era5_temperature", payload["data"]["context_columns"])
        self.assertIn("station_age_years", payload["data"]["continuous_metadata_columns"])

    def test_joint_build_cli_prints_artifacts(self) -> None:
        aws_csv = self.root / "aws.csv"
        noaa_csv = self.root / "noaa.csv"
        aws_csv.write_text(
            "timestamp,station_id,latitude,longitude,temperature,humidity,pressure,wind_speed,wind_direction,precipitation,elevation,cost,sensor_type,sensor_group,sensor_modality,site_type,maintenance_state,maintenance_age,source,raw_timestamp\n",
            encoding="utf-8",
        )
        noaa_csv.write_text(
            "timestamp,station_id,latitude,longitude,air_temperature,dew_point_temperature,sea_level_pressure,wind_speed,wind_direction,precipitation_1hr,elevation,station_age_years,sensor_cost,sensor_type,sensor_group,sensor_modality,site_type,maintenance_state,source,raw_station_name,icao\n",
            encoding="utf-8",
        )
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "joint-build",
                    "--aws-csv",
                    str(aws_csv),
                    "--noaa-csv",
                    str(noaa_csv),
                    "--output-dir",
                    str(self.root / "joint_cli"),
                    "--overwrite",
                ]
            )

        self.assertEqual(code, 0)
        self.assertIn("Built joint dataset", output.getvalue())
        self.assertTrue((self.root / "joint_cli" / "joint_weather_network_q1.csv").exists())


if __name__ == "__main__":
    unittest.main()
