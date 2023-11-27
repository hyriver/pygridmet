"""Access the GridMET database for both single single pixel and gridded queries."""
# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import functools
import io
import itertools
import re
import warnings
from typing import TYPE_CHECKING, Generator, Iterable, Sequence, Union, cast

import numpy as np
import pandas as pd
import shapely
import xarray as xr

import async_retriever as ar
import pygeoogc as ogc
import pygeoutils as geoutils
from pygeoogc import ServiceError, ServiceURL
from pygeoutils import Coordinates
from pygridmet.core import T_RAIN, T_SNOW, GridMET
from pygridmet.exceptions import InputRangeError, InputTypeError

if TYPE_CHECKING:
    from pathlib import Path

    import pyproj
    from shapely import MultiPolygon, Polygon

    CRSTYPE = Union[int, str, pyproj.CRS]

DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"
MAX_CONN = 10

__all__ = ["get_bycoords", "get_bygeom"]


def _coord_urls(
    coord: tuple[float, float],
    variables: Iterable[str],
    dates: list[tuple[pd.Timestamp, pd.Timestamp]],
    long_names: dict[str, str],
) -> Generator[list[tuple[str, dict[str, dict[str, str]]]], None, None]:
    """Generate an iterable URL list for downloading GridMET data.

    Parameters
    ----------
    coord : tuple of length 2
        Coordinates in EPSG:4326 CRS (lon, lat)
    variables : list
        A list of GridMET variables
    dates : list
        A list of dates
    long_names : dict
        A dictionary of long names for the variables.

    Returns
    -------
    generator
        An iterator of generated URLs.
    """
    lon, lat = coord
    return (
        [
            (
                f"{ServiceURL().restful.gridmet}/agg_met_{v}_1979_CurrentYear_CONUS.nc",
                {
                    "params": {
                        "var": long_names[v],
                        "longitude": f"{lon:0.6f}",
                        "latitude": f"{lat:0.6f}",
                        "time_start": s.strftime(DATE_FMT),
                        "time_end": e.strftime(DATE_FMT),
                        "accept": "csv",
                    }
                },
            )
            for s, e in dates
        ]
        for v in variables
    )


def _get_lon_lat(
    coords: list[tuple[float, float]] | tuple[float, float],
    coords_id: Sequence[str | int] | None = None,
    crs: CRSTYPE = 4326,
    to_xarray: bool = False,
) -> tuple[list[float], list[float]]:
    """Get longitude and latitude from a list of coordinates."""
    coords_list = geoutils.coords_list(coords)

    if to_xarray and coords_id is not None and len(coords_id) != len(coords_list):
        raise InputTypeError("coords_id", "list with the same length as of coords")

    coords_list = ogc.match_crs(coords_list, crs, 4326)
    lon, lat = zip(*coords_list)
    return list(lon), list(lat)


def _by_coord(
    lon: float,
    lat: float,
    gridmet: GridMET,
    dates: list[tuple[pd.Timestamp, pd.Timestamp]],
    snow: bool,
    snow_params: dict[str, float] | None,
    ssl: bool,
) -> pd.DataFrame:
    """Get climate data for a coordinate and return as a DataFrame."""
    coords = (lon, lat)
    url_kwds = _coord_urls(coords, gridmet.variables, dates, gridmet.long_names)
    retrieve = functools.partial(ar.retrieve_text, max_workers=MAX_CONN, ssl=ssl)

    clm = pd.concat(
        (
            pd.concat(
                pd.read_csv(io.StringIO(r), parse_dates=[0], index_col=[0], usecols=[0, 3])
                for r in retrieve(u, k)
            )
            for u, k in (zip(*u) for u in url_kwds)
        ),
        axis=1,
    )
    # Rename the columns from their long names to abbreviations and put the units in parentheses
    abbrs = {v: k for k, v in gridmet.long_names.items()}
    clm.columns = clm.columns.str.replace(r'\[unit="(.+)"\]', "", regex=True)
    clm.columns = clm.columns.map(abbrs).map(lambda x: f"{x} ({gridmet.units[x]})")

    clm = clm.set_index(pd.to_datetime(clm.index.strftime("%Y-%m-%d")))
    clm = clm.where(clm < gridmet.missing_value)

    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = gridmet.separate_snow(clm, **params)
    clm.index.name = "time"
    return clm


def get_bycoords(
    coords: list[tuple[float, float]] | tuple[float, float],
    dates: tuple[str, str] | int | list[int],
    coords_id: Sequence[str | int] | None = None,
    crs: CRSTYPE = 4326,
    variables: Iterable[str] | str | None = None,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    ssl: bool = True,
    to_xarray: bool = False,
) -> pd.DataFrame | xr.Dataset:
    """Get point-data from the GridMET database at 1-km resolution.

    This function uses THREDDS data service to get the coordinates
    and supports getting monthly and annual summaries of the climate
    data directly from the server.

    Parameters
    ----------
    coords : tuple or list of tuples
        Coordinates of the location(s) of interest as a tuple (x, y)
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years ``[2001, 2010, ...]``.
    coords_id : list of int or str, optional
        A list of identifiers for the coordinates. This option only applies when ``to_xarray``
        is set to ``True``. If not provided, the coordinates will be enumerated.
    crs : str, int, or pyproj.CRS, optional
        The CRS of the input coordinates, defaults to ``EPSG:4326``.
    variables : str or list
        List of variables to be downloaded. The acceptable variables are:
        ``pr``, ``rmax``, ``rmin``, ``sph``, ``srad``, ``th``, ``tmmn``, ``tmmx``, ``vs``,
        ``bi``, ``fm100``, ``fm1000``, ``erc``, ``etr``, ``pet``, and ``vpd``.
        Descriptions can be found `here <https://www.climatologylab.org/gridmet.html>`__.
        Defaults to ``None``, i.e., all the variables are downloaded.
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0.6}`` that are adopted from
        https://doi.org/10.5194/gmd-11-1077-2018.
    ssl : bool, optional
        Whether to verify SSL certification, defaults to ``True``.
    to_xarray : bool, optional
        Return the data as an ``xarray.Dataset``. Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        Daily climate data for a single or list of locations.

    Examples
    --------
    >>> import pygridmet as gridmet
    >>> coords = (-1431147.7928, 318483.4618)
    >>> dates = ("2000-01-01", "2000-01-31")
    >>> clm = gridmet.get_bycoords(
    ...     coords,
    ...     dates,
    ...     crs=3542,
    ... )
    >>> clm["pr (mm)"].mean()
    9.677

    References
    ----------
    .. footbibliography::
    """
    gridmet = GridMET(variables, snow)
    gridmet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_itr = gridmet.dates_tolist(dates)
    else:
        dates_itr = gridmet.years_tolist(dates)

    lon, lat = _get_lon_lat(coords, coords_id, crs, to_xarray)
    points = Coordinates(lon, lat, gridmet.bounds).points
    n_pts = len(points)
    if n_pts == 0 or n_pts != len(lon):
        raise InputRangeError("coords", f"within {gridmet.bounds}")

    by_coord = functools.partial(
        _by_coord,
        gridmet=gridmet,
        dates=dates_itr,
        snow=snow,
        snow_params=snow_params,
        ssl=ssl,
    )
    clm_list = itertools.starmap(by_coord, zip(points.x, points.y))

    idx = list(coords_id) if coords_id is not None else list(range(n_pts))
    if to_xarray:
        clm_ds = xr.concat(
            (xr.Dataset.from_dataframe(clm) for clm in clm_list), dim=pd.Index(idx, name="id")
        )
        clm_ds = clm_ds.rename(
            {n: re.sub(r"\([^\)]*\)", "", str(n)).strip() for n in clm_ds.data_vars}
        )
        clm_ds["time"] = pd.DatetimeIndex(pd.to_datetime(clm_ds["time"]).date)
        for v in clm_ds.data_vars:
            clm_ds[v].attrs["units"] = gridmet.units[v]
            clm_ds[v].attrs["long_name"] = gridmet.long_names[v]
        clm_ds["lat"] = (("id",), points.y)
        clm_ds["lon"] = (("id",), points.x)
        return clm_ds

    if n_pts == 1:
        clm = next(iter(clm_list), pd.DataFrame())
    else:
        clm = pd.concat(clm_list, keys=idx, axis=1)
        clm = clm.columns.set_names(["id", "variable"])
    clm = clm.set_index(pd.DatetimeIndex(pd.to_datetime(clm.index).date))
    return clm


def _gridded_urls(
    bounds: tuple[float, float, float, float],
    variables: Iterable[str],
    dates: list[tuple[pd.Timestamp, pd.Timestamp]],
    long_names: dict[str, str],
) -> Generator[tuple[str, dict[str, dict[str, str]]], None, None]:
    """Generate an iterable URL list for downloading GridMET data.

    Parameters
    ----------
    bounds : tuple of length 4
        Bounding box (west, south, east, north)
    variables : list
        A list of GridMET variables
    dates : list
        A list of dates
    long_names : dict
        A dictionary of long names for the variables.

    Returns
    -------
    generator
        An iterator of generated URLs.
    """
    west, south, east, north = bounds
    return (
        (
            f"{ServiceURL().restful.gridmet}/agg_met_{v}_1979_CurrentYear_CONUS.nc",
            {
                "params": {
                    "var": long_names[v],
                    "north": f"{north:0.6f}",
                    "west": f"{west:0.6f}",
                    "east": f"{east:0.6f}",
                    "south": f"{south:0.6f}",
                    "disableProjSubset": "on",
                    "horizStride": "1",
                    "time_start": s.strftime(DATE_FMT),
                    "time_end": e.strftime(DATE_FMT),
                    "timeStride": "1",
                    "accept": "netcdf",
                }
            },
        )
        for v, (s, e) in itertools.product(variables, dates)
    )


def _open_dataset(f: Path) -> xr.Dataset:
    """Open a dataset using ``xarray``."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=xr.SerializationWarning)
        with xr.open_dataset(f) as ds:
            return ds.load()


def get_bygeom(
    geometry: Polygon | MultiPolygon | tuple[float, float, float, float],
    dates: tuple[str, str] | int | list[int],
    crs: CRSTYPE = 4326,
    variables: Iterable[str] | str | None = None,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    ssl: bool = True,
) -> xr.Dataset:
    """Get gridded data from the GridMET database at 1-km resolution.

    Parameters
    ----------
    geometry : Polygon, MultiPolygon, or bbox
        The geometry of the region of interest.
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].
    crs : str, int, or pyproj.CRS, optional
        The CRS of the input geometry, defaults to epsg:4326.
    variables : str or list
        List of variables to be downloaded. The acceptable variables are:
        ``pr``, ``rmax``, ``rmin``, ``sph``, ``srad``, ``th``, ``tmmn``, ``tmmx``, ``vs``,
        ``bi``, ``fm100``, ``fm1000``, ``erc``, ``etr``, ``pet``, and ``vpd``.
        Descriptions can be found `here <https://www.climatologylab.org/gridmet.html>`__.
        Defaults to ``None``, i.e., all the variables are downloaded.
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0.6}`` that are adopted from
        https://doi.org/10.5194/gmd-11-1077-2018.
    ssl : bool, optional
        Whether to verify SSL certification, defaults to ``True``.

    Returns
    -------
    xarray.Dataset
        Daily climate data within the target geometry.

    Examples
    --------
    >>> from shapely import Polygon
    >>> import pygridmet as gridmet
    >>> geometry = Polygon(
    ...     [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
    ... )
    >>> clm = gridmet.get_bygeom(geometry, 2010, variables="tmmn")
    >>> clm["tmmn"].mean().item()
    274.167

    References
    ----------
    .. footbibliography::
    """
    gridmet = GridMET(variables, snow)
    gridmet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_itr = gridmet.dates_tolist(dates)
    else:
        dates_itr = gridmet.years_tolist(dates)

    crs = ogc.validate_crs(crs)
    _geometry = geoutils.geo2polygon(geometry, crs, 4326)

    if not _geometry.intersects(shapely.box(*gridmet.bounds)):
        raise InputRangeError("geometry", f"within {gridmet.bounds}")

    urls, kwds = zip(
        *_gridded_urls(
            _geometry.bounds,
            gridmet.variables,
            dates_itr,
            gridmet.long_names,
        )
    )
    urls = cast("list[str]", list(urls))
    kwds = cast("list[dict[str, dict[str, str]]]", list(kwds))

    clm_files = ogc.streaming_download(
        urls,
        kwds,
        file_extention="nc",
        ssl=ssl,
        n_jobs=MAX_CONN,
    )
    try:
        # open_mfdataset can run into too many open files error so we use merge
        # https://docs.xarray.dev/en/stable/user-guide/io.html#reading-multi-file-datasets
        clm = xr.merge(_open_dataset(f) for f in clm_files)
    except ValueError as ex:
        msg = " ".join(
            (
                "GridMET did NOT process your request successfully.",
                "Check your inputs and try again.",
            )
        )
        raise ServiceError(msg) from ex

    clm = xr.where(clm < gridmet.missing_value, clm, np.nan, keep_attrs=True)
    for v in clm.data_vars:
        clm[v] = clm[v].rio.write_nodata(np.nan)
    clm = geoutils.xd_write_crs(clm, 4326, "spatial_ref")
    clm = clm.drop_vars("crs")
    clm = cast("xr.Dataset", clm)
    clm = geoutils.xarray_geomask(clm, _geometry, 4326)
    abbrs = {v: k for k, v in gridmet.long_names.items() if v in clm.data_vars}
    abbrs["day"] = "time"
    clm = clm.rename(abbrs)
    for v in clm.data_vars:
        clm[v].attrs["long_name"] = gridmet.long_names[v]

    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = gridmet.separate_snow(clm, **params)
    return clm
