"""Command-line interface for PyGridMET."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

import click
import geopandas as gpd
import pandas as pd

from pygridmet import pygridmet as gridmet
from pygridmet.exceptions import (
    InputTypeError,
    InputValueError,
    MissingCRSError,
    MissingItemError,
)

if TYPE_CHECKING:
    DFType = TypeVar("DFType", pd.DataFrame, gpd.GeoDataFrame)
    VARS = Literal[
        "pr",
        "rmax",
        "rmin",
        "sph",
        "srad",
        "th",
        "tmmn",
        "tmmx",
        "vs",
        "bi",
        "fm100",
        "fm1000",
        "erc",
        "etr",
        "pet",
        "vpd",
    ]


def parse_snow(target_df: pd.DataFrame) -> pd.DataFrame:
    """Parse the snow dataframe."""
    if target_df["snow"].dtype != bool:
        target_df["snow"] = target_df.snow.str.lower().str.strip()
        if not target_df.snow.str.contains("true|false").all():
            raise InputValueError("snow", "true or false")
        target_df["snow"] = target_df.snow == "true"
    return target_df


def get_target_df(tdf: DFType, req_cols: list[str]) -> DFType:
    """Check if all required columns exists in the dataframe.

    It also re-orders the columns based on req_cols order.
    """
    missing = [c for c in req_cols if c not in tdf]
    if missing:
        raise MissingItemError(missing)
    return tdf[req_cols]  # pyright: ignore[reportReturnType]


def get_required_cols(geom_type: str, columns: pd.Index) -> list[str]:
    """Get the required columns for a given geometry type."""
    req_cols = ["id", geom_type, "dates"]
    return req_cols + list({"snow"}.intersection(columns))


variables_opt = click.option(
    "--variables",
    "-v",
    multiple=True,
    default=["all"],
    help="Target variables. You can pass this flag multiple times for multiple variables.",
)

years_opt = click.option(
    "--years",
    "-y",
    type=int,
    multiple=True,
    required=True,
    help="Target years. You can pass this flag multiple times for multiple years.",
)

save_dir_opt = click.option(
    "-s",
    "--save_dir",
    type=click.Path(exists=False),
    default="clm_gridmet",
    help=" ".join(
        (
            "Path to a directory to save the requested files.",
            "Extension for the outputs is .nc for geometry and .csv for coords.",
        )
    ),
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
def cli() -> None:
    """Command-line interface for Pygridmet."""


@cli.command("coords", context_settings=CONTEXT_SETTINGS)
@click.argument("fpath", type=click.Path(exists=True))
@variables_opt
@save_dir_opt
def coords(
    fpath: Path,
    variables: list[VARS] | VARS | Literal["all"] = "all",
    save_dir: str | Path = "clm_gridmet",
) -> None:
    """Retrieve climate data for a list of coordinates.

    \b
    FPATH: Path to a csv file with four columns:
        - ``id``: Feature identifiers that gridmet uses as the output netcdf filenames.
        - ``start``: Start time.
        - ``end``: End time.
        - ``lon``: Longitude of the points of interest.
        - ``lat``: Latitude of the points of interest.
        - ``snow``: (optional) Separate snowfall from precipitation, default is ``False``.

    \b
    Examples:
        $ cat coords.csv
        id,lon,lat,start,end
        california,-122.2493328,37.8122894,2012-01-01,2014-12-31
        $ pygridmet coords coords.csv -v prcp -v tmin
    """  # noqa: D301
    fpath = Path(fpath)
    if fpath.suffix != ".csv":
        raise InputTypeError("file", ".csv")

    target_df = get_target_df(pd.read_csv(fpath), ["id", "start", "end", "lon", "lat"])
    target_df["dates"] = list(target_df[["start", "end"]].itertuples(index=False, name=None))
    target_df["coords"] = list(target_df[["lon", "lat"]].itertuples(index=False, name=None))
    if "snow" in target_df:
        target_df = parse_snow(target_df)

    req_cols = get_required_cols("coords", target_df.columns)
    target_df = target_df[req_cols]

    count = "1 point" if len(target_df) == 1 else f"{len(target_df)} points"
    click.echo(f"Found coordinates of {count} in {fpath.resolve()}.")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with click.progressbar(
        target_df.itertuples(index=False, name=None),
        label="Getting single-pixel climate data",
        length=len(target_df),
    ) as bar:
        for i, *args in bar:
            fname = Path(save_dir, f"{i}.csv")
            if fname.exists():
                continue
            kwrgs = dict(zip(req_cols[1:], args))
            clm = gridmet.get_bycoords(
                **kwrgs,
                variables=None if "all" in variables else variables,
            )
            clm.to_csv(fname, index=False)
    click.echo("Done.")


@cli.command("geometry", context_settings=CONTEXT_SETTINGS)
@click.argument("fpath", type=click.Path(exists=True))
@variables_opt
@save_dir_opt
def geometry(
    fpath: Path,
    variables: list[VARS] | VARS | Literal["all"] = "all",
    save_dir: str | Path = "clm_gridmet",
) -> None:
    """Retrieve climate data for a dataframe of geometries.

    \b
    FPATH: Path to a shapefile (.shp) or geopackage (.gpkg) file.
    This file must have four columns and contain a ``crs`` attribute:
        - ``id``: Feature identifiers that gridmet uses as the output netcdf filenames.
        - ``start``: Start time.
        - ``end``: End time.
        - ``geometry``: Target geometries.
        - ``snow``: (optional) Separate snowfall from precipitation, default is ``False``.

    \b
    Examples:
        $ pygridmet geometry geo.gpkg -v prcp -v tmin
    """  # noqa: D301
    fpath = Path(fpath)
    if fpath.suffix not in (".shp", ".gpkg"):
        raise InputTypeError("file", ".shp or .gpkg")

    target_df = gpd.read_file(fpath)
    if target_df.crs is None:
        raise MissingCRSError

    if "undefined geographic" in target_df.crs.name.lower():
        raise MissingCRSError

    target_df = get_target_df(target_df, ["id", "start", "end", "geometry"])
    target_df["dates"] = list(target_df[["start", "end"]].itertuples(index=False, name=None))
    req_cols = get_required_cols("geometry", target_df.columns)
    target_df = target_df[req_cols]

    count = "1 geometry" if len(target_df) == 1 else f"{len(target_df)} geometries"
    click.echo(f"Found {count} in {fpath.resolve()}.")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with click.progressbar(
        target_df.itertuples(index=False, name=None),
        label="Getting gridded climate data",
        length=len(target_df),
    ) as bar:
        for i, *args in bar:
            fname = Path(save_dir, f"{i}.nc")
            if fname.exists():
                continue
            kwrgs = dict(zip(req_cols[1:], args))
            clm = gridmet.get_bygeom(
                **kwrgs,
                crs=target_df.crs,
                variables=None if "all" in variables else variables,
            )
            clm.to_netcdf(fname)
    click.echo("Done.")


@cli.command("conus", context_settings=CONTEXT_SETTINGS)
@years_opt
@variables_opt
@save_dir_opt
def conus(
    years: int | list[int],
    variables: list[VARS] | VARS | Literal["all"] = "all",
    save_dir: str | Path = "clm_gridmet",
) -> None:
    r"""Retrieve climate data for the contiguous United States.

    \b
    Examples:
        $ pygridmet conus -y 2010 -v tmmn
    """
    n_years = f"{len(years)} years" if len(years) > 1 else "1 year"
    n_vars = f"{len(variables)} variables" if len(variables) > 1 else "1 variable"
    click.echo(f"Getting CONUS data for {n_years} and {n_vars} ...")
    gridmet.get_conus(
        years,
        variables=None if "all" in variables else variables,
        save_dir=save_dir,
    )
    click.echo("Done.")
