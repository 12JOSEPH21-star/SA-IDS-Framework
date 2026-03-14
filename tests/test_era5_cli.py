from __future__ import annotations

import io
import importlib.util
import json
import tempfile
import unittest
import zipfile
from csv import DictReader
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

from task_cli.app import main
from task_cli.era5 import Era5DownloadConfig, build_download_plan, convert_era5_to_reference_csv


XARRAY_AVAILABLE = importlib.util.find_spec("xarray") is not None
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class Era5CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_build_download_plan_splits_range_by_month(self) -> None:
        config = Era5DownloadConfig(
            start_date=date(2025, 1, 30),
            end_date=date(2025, 2, 2),
            output_dir=self.root / "era5",
            csv_path=self.root / "era5" / "era5_reference.csv",
            area=(39.0, 124.0, 32.0, 132.0),
            grid=(0.25, 0.25),
            dry_run=True,
        )

        plan = build_download_plan(config)

        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[0].kind, "static")
        self.assertEqual(plan[1].request["day"], ["30", "31"])
        self.assertEqual(plan[2].request["day"], ["01", "02"])
        self.assertEqual(plan[1].request["area"], [39.0, 124.0, 32.0, 132.0])
        self.assertEqual(plan[1].request["grid"], [0.25, 0.25])

    def test_era5_download_dry_run_writes_manifest_and_framework_config(self) -> None:
        output_dir = self.root / "data" / "era5"
        csv_path = output_dir / "era5_reference.csv"
        framework_config = self.root / "framework_era5.json"

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "era5-download",
                    "--start-date",
                    "2025-01-30",
                    "--end-date",
                    "2025-02-02",
                    "--output-dir",
                    str(output_dir),
                    "--csv-path",
                    str(csv_path),
                    "--framework-config",
                    str(framework_config),
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)
        manifest_path = output_dir / "era5_download_manifest.json"
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["dry_run"])
        self.assertEqual(manifest["auth_source"], "missing")
        self.assertEqual(len(manifest["tasks"]), 3)
        self.assertEqual(manifest["tasks"][1]["request"]["day"], ["30", "31"])
        self.assertEqual(manifest["tasks"][2]["request"]["day"], ["01", "02"])
        self.assertEqual(Path(manifest["csv_path"]), csv_path.resolve())

        self.assertTrue(framework_config.exists())
        framework_payload = json.loads(framework_config.read_text(encoding="utf-8"))
        self.assertEqual(framework_payload["preset"], "era5_reference")
        self.assertEqual(framework_payload["data"]["target_column"], "t2m")
        self.assertIn("Planned 2 monthly ERA5 requests", output.getvalue())
        self.assertIn("Framework config", output.getvalue())

    def test_era5_download_dry_run_accepts_inline_credentials(self) -> None:
        output_dir = self.root / "data" / "era5"

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "era5-download",
                    "--start-date",
                    "2025-01-01",
                    "--end-date",
                    "2025-01-01",
                    "--output-dir",
                    str(output_dir),
                    "--no-csv",
                    "--dry-run",
                    "--cds-url",
                    "https://cds.climate.copernicus.eu/api",
                    "--cds-key",
                    "00000:example-token",
                ]
            )

        self.assertEqual(code, 0)
        manifest = json.loads((output_dir / "era5_download_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["auth_source"], "cli")
        self.assertEqual(len(manifest["tasks"]), 2)

    def test_era5_download_rejects_framework_config_without_csv(self) -> None:
        output_dir = self.root / "data" / "era5"
        framework_config = self.root / "framework_era5.json"

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "era5-download",
                    "--start-date",
                    "2025-01-01",
                    "--end-date",
                    "2025-01-01",
                    "--output-dir",
                    str(output_dir),
                    "--framework-config",
                    str(framework_config),
                    "--no-csv",
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("Framework config generation requires CSV conversion", output.getvalue())
        self.assertFalse(framework_config.exists())

    def test_era5_download_sparse_plan_uses_row_split_for_framework_config(self) -> None:
        output_dir = self.root / "data" / "era5"
        framework_config = self.root / "framework_era5.json"

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "era5-download",
                    "--start-date",
                    "2025-01-01",
                    "--end-date",
                    "2025-01-01",
                    "--output-dir",
                    str(output_dir),
                    "--framework-config",
                    str(framework_config),
                    "--hours",
                    "00:00",
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)
        payload = json.loads(framework_config.read_text(encoding="utf-8"))
        self.assertEqual(payload["data"]["split_strategy"], "row")
        self.assertIn("Framework config", output.getvalue())

    def test_era5_download_reports_missing_cdsapi_dependency(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(
                [
                    "era5-download",
                    "--start-date",
                    "2025-01-01",
                    "--end-date",
                    "2025-01-01",
                    "--output-dir",
                    str(self.root / "data" / "era5"),
                    "--no-csv",
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("cdsapi", output.getvalue())

    @unittest.skipUnless(XARRAY_AVAILABLE and PANDAS_AVAILABLE, "requires xarray and pandas")
    def test_convert_era5_reference_csv_reads_zip_containers(self) -> None:
        import xarray as xr

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as ascii_temp:
            work = Path(ascii_temp)
            dynamic_zip = work / "dynamic.nc"
            static_nc = work / "static.nc"
            csv_path = work / "era5_reference.csv"

            instant = xr.Dataset(
                data_vars={
                    "t2m": (("time", "latitude", "longitude"), [[[280.0]]]),
                    "u10": (("time", "latitude", "longitude"), [[[1.0]]]),
                    "v10": (("time", "latitude", "longitude"), [[[2.0]]]),
                    "sp": (("time", "latitude", "longitude"), [[[101325.0]]]),
                },
                coords={"time": ["2025-01-01T00:00:00"], "latitude": [35.0], "longitude": [129.0]},
            )
            accum = xr.Dataset(
                data_vars={"tp": (("time", "latitude", "longitude"), [[[0.001]]])},
                coords={"time": ["2025-01-01T00:00:00"], "latitude": [35.0], "longitude": [129.0]},
            )
            instant_nc = work / "instant.nc"
            accum_nc = work / "accum.nc"
            instant.to_netcdf(instant_nc)
            accum.to_netcdf(accum_nc)
            with zipfile.ZipFile(dynamic_zip, "w") as archive:
                archive.write(instant_nc, arcname="instant.nc")
                archive.write(accum_nc, arcname="accum.nc")

            static = xr.Dataset(
                data_vars={
                    "z": (("latitude", "longitude"), [[98.0665]]),
                    "lsm": (("latitude", "longitude"), [[1.0]]),
                },
                coords={"latitude": [35.0], "longitude": [129.0]},
            )
            static.to_netcdf(static_nc)

            output_path = convert_era5_to_reference_csv((dynamic_zip,), csv_path, static_path=static_nc)

            text = output_path.read_text(encoding="utf-8")
            self.assertIn("grid_id", text)
            self.assertIn("t2m", text)
            self.assertIn("orography", text)
            self.assertIn("era5_35.000_129.000", text)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                row = next(DictReader(handle))
            self.assertAlmostEqual(float(row["orography"]), 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
