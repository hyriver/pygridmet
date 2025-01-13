"""Tests for PyDaymet package."""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely import Polygon

import pygridmet as gridmet
from pygridmet.cli import cli

GEOM = Polygon(
    [
        [-69.77, 45.07],
        [-69.31, 45.07],
        [-69.31, 45.45],
        [-69.77, 45.45],
        [-69.77, 45.07],
    ]
)
DAY = ("2000-01-01", "2000-01-12")
YEAR = 2010
VAR = ["pr", "tmmn"]
DEF_CRS = 4326
ALT_CRS = 3542
COORDS = (-1431147.7928, 318483.4618)
DATES = ("2000-01-01", "2000-12-31")


def assert_close(a: float, b: float, rtol: float = 1e-3) -> bool:
    assert np.isclose(a, b, rtol=rtol).all()


class TestByCoords:
    @pytest.mark.jit
    def test_snow(self):
        clm = gridmet.get_bycoords(COORDS, DATES, snow=True, crs=ALT_CRS)
        assert_close(clm["snow (mm)"].mean(), 0.0)

    def test_daily(self):
        clm = gridmet.get_bycoords(COORDS, DATES, variables=VAR, crs=ALT_CRS)
        clm_ds = gridmet.get_bycoords(COORDS, DATES, variables=VAR, crs=ALT_CRS, to_xarray=True)

        expected = 8.8493
        assert_close(clm["pr (mm)"].mean(), expected)
        assert_close(clm_ds.pr.mean(), expected)


class TestByGeom:
    @pytest.mark.jit
    def test_snow(self):
        clm = gridmet.get_bygeom(GEOM, DAY, snow=True, snow_params={"t_snow": 0.5})
        assert_close(clm.snow.mean().item(), 3.4895)

    def test_bounds(self):
        clm = gridmet.get_bygeom(GEOM.bounds, DAY)
        assert_close(clm.pr.mean().item(), 3.4895)

    def test_daily(self):
        clm = gridmet.get_bygeom(GEOM, DAY, variables=VAR)
        assert_close(clm.tmmn.mean().item(), 264.0151)


class TestCLI:
    """Test the command-line interface."""

    def test_geometry(self, runner):
        import geopandas as gpd

        params = {
            "id": "geo_test",
            "start": "2000-01-01",
            "end": "2000-05-31",
            "snow": "false",
        }
        geo_gpkg = Path("nat_geo.gpkg")
        save_dir = "test_geometry"
        gdf = gpd.GeoDataFrame(params, geometry=[GEOM], index=[0], crs=DEF_CRS)
        gdf.to_file(geo_gpkg)
        ret = runner.invoke(
            cli,
            [
                "geometry",
                str(geo_gpkg),
                *[item for v in VAR for item in ["-v", v]],
                "-s",
                save_dir,
            ],
        )
        if geo_gpkg.is_dir():
            shutil.rmtree(geo_gpkg)
        else:
            geo_gpkg.unlink()
        shutil.rmtree(save_dir, ignore_errors=True)
        assert str(ret.exception) == "None"
        assert ret.exit_code == 0
        assert "Found 1 geometry" in ret.output

    @pytest.mark.jit
    def test_coords(self, runner):
        params = {
            "id": "coords_test",
            "lon": -69.77,
            "lat": 45.07,
            "start": DAY[0],
            "end": DAY[1],
            "snow": "TRUE",
        }
        coord_csv = "coords.csv"
        save_dir = "test_coords"
        df = pd.DataFrame(params, index=[0])
        df.to_csv(coord_csv, index=False)
        ret = runner.invoke(
            cli,
            [
                "coords",
                coord_csv,
                *[item for v in VAR for item in ["-v", v]],
                "-s",
                save_dir,
            ],
        )
        runner.invoke(
            cli,
            [
                "coords",
                coord_csv,
                *[item for v in VAR for item in ["-v", v]],
                "-s",
                save_dir,
            ],
        )
        Path(coord_csv).unlink()
        shutil.rmtree(save_dir, ignore_errors=True)
        assert str(ret.exception) == "None"
        assert ret.exit_code == 0
        assert "Found coordinates of 1 point" in ret.output


def test_conus(runner):
    fnames = gridmet.get_conus(2010, variables="tmmn")
    assert fnames[0].exists()

    save_dir = "test_conus"
    ret = runner.invoke(cli, ["conus", "-y", 2010, "-v", "tmmn", "-s", save_dir])
    assert Path(save_dir, "tmmn_2010.nc").exists()
    shutil.rmtree(save_dir, ignore_errors=True)
    assert str(ret.exception) == "None"
    assert ret.exit_code == 0
    assert "Getting CONUS data for 1 year and 1 variable" in ret.output


def test_show_versions():
    f = io.StringIO()
    gridmet.show_versions(file=f)
    assert "SYS INFO" in f.getvalue()
