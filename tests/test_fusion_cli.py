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

    def test_build_joint_dataset_supports_asos_event_qc_and_nwp_enrichment(self) -> None:
        aws_csv = self.root / "aws.csv"
        aws_csv.write_text(
            "\n".join(
                [
                    "timestamp,station_id,latitude,longitude,temperature,humidity,pressure,wind_speed,wind_direction,precipitation,elevation,cost,sensor_type,sensor_group,sensor_modality,site_type,maintenance_state,maintenance_age,source,raw_timestamp",
                    "2026-03-16T00:00,90,38.25,128.56,0.1,50.0,1019.0,1.5,255.8,0.0,17.5,,aws,surface,automatic_weather_station,coastal,nominal,1.0,aws_recent_1min,202603160000",
                ]
            ),
            encoding="utf-8",
        )
        asos_csv = self.root / "asos.csv"
        asos_csv.write_text(
            "\n".join(
                [
                    "timestamp,station_id,latitude,longitude,temperature,humidity,pressure,wind_speed,wind_direction,precipitation,elevation,cost,sensor_type,sensor_group,sensor_modality,site_type,maintenance_state,maintenance_age,source,raw_timestamp",
                    "2026-03-16T00:00,108,37.57,126.97,2.0,60.0,1018.0,1.1,180.0,0.0,85.0,,asos,surface,synoptic_surface,urban,nominal,2.0,asos_hourly_apihub,202603160000",
                ]
            ),
            encoding="utf-8",
        )
        noaa_csv = self.root / "noaa.csv"
        noaa_csv.write_text(
            "\n".join(
                [
                    "timestamp,station_id,latitude,longitude,air_temperature,dew_point_temperature,sea_level_pressure,wind_speed,wind_direction,precipitation_1hr,elevation,station_age_years,sensor_cost,sensor_type,sensor_group,sensor_modality,site_type,maintenance_state,source,raw_station_name,icao",
                    "2026-03-16T01:00,471080-99999,37.57,126.97,-1.0,-3.0,1020.0,2.0,180,0.0,86.0,20.0,1.0,isd_station,surface,synoptic_surface,airport,,noaa_isd_lite,SEOUL STATION,RKSS",
                ]
            ),
            encoding="utf-8",
        )
        event_history_csv = self.root / "warning_history_standardized.csv"
        event_history_csv.write_text(
            "\n".join(
                [
                    "source,basis,publication_time,effective_time,input_time,start_time,end_time,region_id,region_parent_id,region_short_name,region_name,issuing_office,issuing_station_id,warning_code,warning_level,command,grade,status_count,report_flag,sequence,forecaster,operator,raw_text",
                    "warning_history,f,2026-03-16T00:00,2026-03-16T00:00,,2026-03-16T00:00,2026-03-16T02:00,0,0,ALL,ALL,KMA,108,R,2,,,,,,,heavy rain",
                ]
            ),
            encoding="utf-8",
        )
        qc_csv = self.root / "qc.csv"
        qc_csv.write_text(
            "\n".join(
                [
                    "source,station_id,hour_bucket_availability_ratio,status_or_qc_flag_count,suspect_value_count,raw_rows_per_observed_hour",
                    "aws,90,0.9,1,2,4.0",
                    "asos,108,0.8,0,1,1.0",
                ]
            ),
            encoding="utf-8",
        )
        station_event_csv = self.root / "station_event.csv"
        station_event_csv.write_text(
            "\n".join(
                [
                    "source,station_id,station_admin,hour_timestamp,event_station_active,event_station_count,event_station_max_warning_level,first_event_time,last_input_time",
                    "aws,90,105,2026-03-16T00:00,1.0,2,3,2026-03-16T00:03,2026-03-16T00:12",
                ]
            ),
            encoding="utf-8",
        )
        nwp_summary_csv = self.root / "ldaps_summary.csv"
        raw_dir = self.root / "raw"
        raw_dir.mkdir()
        raw_json_path = raw_dir / "ldaps_unis_all_202603160000_h000.json"
        raw_json_path.write_text(
            json.dumps(
                {
                    "response": {
                        "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
                        "body": {
                            "items": {
                                "item": [
                                    {
                                        "baseTime": "202603160000",
                                        "fcstTime": "202603160000",
                                        "dataTypeCd": "Temp",
                                        "value": "10.0,20.0",
                                    }
                                ]
                            }
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        nwp_summary_csv.write_text(
            "\n".join(
                [
                    "source,base_time,forecast_time,lead_hour,data_type_code,grid_km,xdim,ydim,x0,y0,unit,lon,lat,item_index,item_count,value_count,value_preview,raw_path",
                    f"ldaps_unis_all,2026-03-16T00:00,2026-03-16T00:00,0,Temp,1.5,1,2,0,0,C,,,0,1,2,10.0,{raw_json_path}",
                ]
            ),
            encoding="utf-8",
        )
        nwp_grid_csv = self.root / "ldaps_grid.csv"
        nwp_grid_csv.write_text(
            "\n".join(
                [
                    "nwp_code,coordinate_type,grid_index,value",
                    "u015,lon,0,127.00",
                    "u015,lat,0,37.00",
                    "u015,lon,1,128.56",
                    "u015,lat,1,38.25",
                ]
            ),
            encoding="utf-8",
        )

        artifacts = build_joint_dataset(
            JointBuildConfig(
                aws_csv_path=aws_csv,
                asos_csv_path=asos_csv,
                noaa_csv_path=noaa_csv,
                event_history_csv_path=event_history_csv,
                event_station_csv_path=station_event_csv,
                qc_metadata_csv_path=qc_csv,
                nwp_ldaps_summary_csv_path=nwp_summary_csv,
                nwp_ldaps_grid_csv_path=nwp_grid_csv,
                output_dir=self.root / "joint_enriched",
            )
        )

        with artifacts.output_csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_source = {row["source"]: row for row in rows}
        self.assertEqual(by_source["asos"]["event_warning_active"], "1.0")
        self.assertEqual(by_source["aws"]["ldaps_temperature"], "20.0")
        self.assertEqual(by_source["aws"]["qc_hour_bucket_availability_ratio"], "0.9")
        self.assertEqual(by_source["aws"]["event_station_active"], "1.0")
        self.assertEqual(by_source["aws"]["event_station_count"], "2")

    def test_joint_framework_config_falls_back_when_nwp_has_no_overlap(self) -> None:
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
            "timestamp,station_id,latitude,longitude,air_temperature,dew_point_temperature,sea_level_pressure,wind_speed,wind_direction,precipitation_1hr,elevation,station_age_years,sensor_cost,sensor_type,sensor_group,sensor_modality,site_type,maintenance_state,source,raw_station_name,icao\n",
            encoding="utf-8",
        )
        nwp_summary_csv = self.root / "ldaps_summary.csv"
        raw_dir = self.root / "raw"
        raw_dir.mkdir()
        raw_json_path = raw_dir / "ldaps_unis_all_202603160000_h000.json"
        raw_json_path.write_text(
            json.dumps(
                {
                    "response": {
                        "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
                        "body": {
                            "items": {
                                "item": [
                                    {
                                        "baseTime": "202603160000",
                                        "fcstTime": "202603160000",
                                        "dataTypeCd": "Temp",
                                        "value": "10.0,20.0",
                                    }
                                ]
                            }
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        nwp_summary_csv.write_text(
            "\n".join(
                [
                    "source,base_time,forecast_time,lead_hour,data_type_code,grid_km,xdim,ydim,x0,y0,unit,lon,lat,item_index,item_count,value_count,value_preview,raw_path",
                    f"ldaps_unis_all,2026-03-16T00:00,2026-03-16T00:00,0,Temp,1.5,1,2,0,0,C,,,0,1,2,10.0,{raw_json_path}",
                ]
            ),
            encoding="utf-8",
        )
        nwp_grid_csv = self.root / "ldaps_grid.csv"
        nwp_grid_csv.write_text(
            "\n".join(
                [
                    "nwp_code,coordinate_type,grid_index,value",
                    "u015,lon,0,127.00",
                    "u015,lat,0,37.00",
                    "u015,lon,1,128.56",
                    "u015,lat,1,38.25",
                ]
            ),
            encoding="utf-8",
        )
        framework_config = self.root / "joint" / "framework_joint.json"

        build_joint_dataset(
            JointBuildConfig(
                aws_csv_path=aws_csv,
                noaa_csv_path=noaa_csv,
                nwp_ldaps_summary_csv_path=nwp_summary_csv,
                nwp_ldaps_grid_csv_path=nwp_grid_csv,
                output_dir=self.root / "joint",
                framework_config_path=framework_config,
            )
        )

        payload = json.loads(framework_config.read_text(encoding="utf-8"))
        self.assertEqual(payload["pipeline"]["observation"]["diagnosis_mode"], "temporal")
        self.assertNotIn("ldaps_temperature", payload["data"]["context_columns"])


if __name__ == "__main__":
    unittest.main()
