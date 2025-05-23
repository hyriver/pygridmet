"""Access the GridMET database for both single single pixel and gridded queries."""

from __future__ import annotations

import itertools
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import shapely
import xarray as xr

import pygridmet._utils as utils
from pygridmet.core import GM_VARS, T_RAIN, T_SNOW, GridMET
from pygridmet.exceptions import InputRangeError, InputTypeError, ServiceError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray
    from pyproj import CRS
    from shapely import Polygon

    CRSType = int | str | CRS
    GMVars = Literal[
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

DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"
N_RETRIES = 15
URL = "http://thredds.northwestknowledge.net:8080/thredds/ncss"

__all__ = ["get_bycoords", "get_bygeom", "get_conus"]


def _coords_to_bounds(
    coords: list[tuple[float, float]] | tuple[float, float],
    bounds: tuple[float, float, float, float],
    coords_id: Sequence[str | int] | None,
    crs: CRSType,
    to_xarray: bool,
) -> tuple[list[tuple[float, float]], NDArray[np.float64]]:
    """Get longitude and latitude from a list of coordinates."""
    coords_list = utils.transform_coords(coords, crs, 4326)

    if to_xarray and coords_id is not None and len(coords_id) != len(coords_list):
        raise InputTypeError("coords_id", "list with the same length as of coords")

    points = shapely.points(utils.validate_coords(coords_list, bounds))
    return coords_list, shapely.bounds(shapely.buffer(points, 0.05)).round(6)


def _gridded_urls(
    bounds: tuple[float, float, float, float],
    variables: Iterable[GMVars],
    dates: list[tuple[pd.Timestamp, pd.Timestamp]],
    long_names: dict[str, str],
) -> list[str]:
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
    list
        A list of generated URLs.
    """
    west, south, east, north = bounds
    return [
        f"{URL}/agg_met_{v}_1979_CurrentYear_CONUS.nc?"
        + urlencode(
            {
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
                "addLatLon": "true",
                "accept": "netcdf",
            }
        )
        for v, (s, e) in itertools.product(variables, dates)
    ]


def _open_dataset(f: Path) -> xr.Dataset:
    """Open a dataset using ``xarray``."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=xr.SerializationWarning)
        with xr.open_dataset(f) as ds:
            return ds.load()


def _check_nans(
    clm: xr.Dataset,
    urls: list[str],
    clm_files: list[Path],
    long2abbr: dict[str, str],
) -> tuple[bool, list[str]]:
    """Check for NaNs, remove files containing NaNs, and return URLs with NaN results."""
    nans = [p for p, v in clm.isnull().sum().any().items() if v.item()]
    if nans:
        nans = [long2abbr[str(n)] for n in nans]
        urls, clm_files = zip(
            *((u, f) for u, f in zip(urls, clm_files) if utils.find_var(u) in nans)
        )
        _ = [f.unlink() for f in clm_files]
        return True, urls
    return False, urls


def _download_urls(
    urls: list[str],
    long2abbr: dict[str, str],
    validate_filesize: bool,
    conn_timeout: int,
) -> xr.Dataset:
    """Download the URLs and return the dataset."""
    clm_all_files = utils.download_files(urls, "nc", None, validate_filesize, conn_timeout)
    clm_files = clm_all_files.copy()
    clm = None
    # Sometimes the server returns NaNs, so we must check for that, remove
    # the files containing NaNs, and try again.
    for _ in range(N_RETRIES):
        try:
            # open_mfdataset can run into too many open files error so we use merge
            # https://docs.xarray.dev/en/stable/user-guide/io.html#reading-multi-file-datasets
            clm = xr.merge(_open_dataset(f) for f in clm_all_files).astype("f4")
        except ValueError:
            _ = [f.unlink() for f in clm_files]
            clm_files = utils.download_files(urls, "nc", None, validate_filesize, conn_timeout)
            clm = None
            continue

        has_nans, urls = _check_nans(clm, urls, clm_files, long2abbr)
        if has_nans:
            clm_files = utils.download_files(urls, "nc", None, validate_filesize, conn_timeout)
            clm = None
            continue
        break

    if clm is None:
        msg = " ".join(
            (
                "GridMET did NOT process your request successfully.",
                "Check your inputs and try again.",
            )
        )
        raise ServiceError(msg)
    return clm


def get_bygeom(
    geometry: Polygon | tuple[float, float, float, float],
    dates: tuple[str, str] | int | list[int],
    crs: CRSType = 4326,
    variables: Iterable[GMVars] | GMVars | None = None,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    conn_timeout: int = 1000,
    validate_filesize: bool = True,
) -> xr.Dataset:
    """Get gridded data from the GridMET database at 1-km resolution.

    Parameters
    ----------
    geometry : Polygon or tuple
        The geometry of the region of interest. It can be a shapely Polygon or a tuple
        of length 4 representing the bounding box (minx, miny, maxx, maxy).
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
    conn_timeout : int, optional
        Connection timeout in seconds, defaults to 1000.
    validate_filesize : bool, optional
        When set to ``True``, the function checks the file size of the previously
        cached files and will re-download if the local filesize does not match
        that of the remote. Defaults to ``True``. Setting this to ``False``
        can be useful when you are sure that the cached files are not corrupted and just
        want to get the combined dataset more quickly. This is faster because it avoids
        web requests that are necessary for getting the file sizes.

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
    """
    gridmet = GridMET(dates, variables, snow)

    crs = utils.validate_crs(crs)
    _geometry = utils.to_geometry(geometry, crs, 4326)
    if not _geometry.intersects(shapely.box(*gridmet.bounds)):
        raise InputRangeError("geometry", f"within {gridmet.bounds}")

    urls = _gridded_urls(
        _geometry.bounds,  # pyright: ignore[reportGeneralTypeIssues]
        gridmet.variables,  # pyright: ignore[reportArgumentType]
        gridmet.date_iterator,
        gridmet.long_names,
    )

    long2abbr = {v: k for k, v in gridmet.long_names.items()}
    clm = _download_urls(urls, long2abbr, validate_filesize, conn_timeout)
    clm = xr.where(clm < gridmet.missing_value, clm, np.nan, keep_attrs=True)

    for v in clm.data_vars:
        _ = clm[v].attrs.pop("grid_mapping", None)
        clm[v] = clm[v].rio.write_nodata(np.nan)
    clm = clm.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs(4326)
    clm = cast("xr.Dataset", clm)
    clm = utils.clip_dataset(clm, _geometry, 4326)
    abbrs = {v: k for k, v in gridmet.long_names.items() if v in clm}
    abbrs["day"] = "time"
    clm = clm.rename(abbrs)
    for v in clm.data_vars:
        clm[v].attrs["long_name"] = gridmet.long_names[v]

    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = gridmet.separate_snow(clm, **params)
    return clm


def get_bycoords(
    coords: list[tuple[float, float]] | tuple[float, float],
    dates: tuple[str, str] | int | list[int],
    coords_id: Sequence[str | int] | None = None,
    crs: CRSType = 4326,
    variables: Iterable[GMVars] | GMVars | None = None,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
    to_xarray: bool = False,
    conn_timeout: int = 1000,
    validate_filesize: bool = True,
) -> pd.DataFrame | xr.Dataset:
    """Get point-data from the GridMET database at 1-km resolution.

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
    to_xarray : bool, optional
        Return the data as an ``xarray.Dataset``. Defaults to ``False``.
    conn_timeout : int, optional
        Connection timeout in seconds, defaults to 1000.
    validate_filesize : bool, optional
        When set to ``True``, the function checks the file size of the previously
        cached files and will re-download if the local filesize does not match
        that of the remote. Defaults to ``True``. Setting this to ``False``
        can be useful when you are sure that the cached files are not corrupted and just
        want to get the combined dataset more quickly. This is faster because it avoids
        web requests that are necessary for getting the file sizes.

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
    """
    gridmet = GridMET(dates, variables, snow)

    coords_list, bounds = _coords_to_bounds(coords, gridmet.bounds, coords_id, crs, to_xarray)
    n_pts = len(coords_list)

    idx = list(coords_id) if coords_id is not None else range(n_pts)
    names = list(gridmet.variables)
    if snow:
        names = [*names, "snow"]
    name_mapping = {name: f"{name} ({gridmet.units[name]})" for name in names}
    clm_dict = {}
    for i, (lon, lat), bbox in zip(idx, coords_list, bounds):
        ds = get_bygeom(
            bbox, dates, crs, variables, snow, snow_params, conn_timeout, validate_filesize
        )
        ds = ds.sel(lon=round(lon, 6), lat=round(lat, 6), method="nearest")
        df = ds.to_dataframe()[names].rename(columns=name_mapping)
        clm_dict[i] = df

    if to_xarray:
        clm_ds = xr.concat(
            (xr.Dataset.from_dataframe(clm) for clm in clm_dict.values()),
            dim=pd.Index(list(clm_dict), name="id"),
        )
        clm_ds = clm_ds.rename(
            {n: re.sub(r"\([^\)]*\)", "", str(n)).strip() for n in clm_ds.data_vars}
        )
        for v in clm_ds.data_vars:
            clm_ds[v].attrs["units"] = gridmet.units[v]
            clm_ds[v].attrs["long_name"] = gridmet.long_names[v]
        lons, lats = zip(*coords_list)
        clm_ds["lat"] = (("id",), list(lats))
        clm_ds["lon"] = (("id",), list(lons))
        return clm_ds

    if n_pts == 1:
        return next(iter(clm_dict.values()), pd.DataFrame())
    return pd.concat(clm_dict.values(), keys=list(clm_dict), axis=1, names=["id", "variable"])


def get_conus(
    years: int | list[int],
    variables: GMVars | list[GMVars] | None = None,
    save_dir: str | Path = "clm_gridmet",
    conn_timeout: int = 1000,
    validate_filesize: bool = True,
) -> list[Path | None]:
    """Get the entire CONUS data for the specified years and variables.

    Parameters
    ----------
    years : int or list
        The year(s) of interest.
    variables : str or list, optional
        The variable(s) of interest, defaults to ``None`` which downloads
        all the variables.
    save_dir : str or Path, optional
        The directory to store the downloaded data, defaults to ``./clm_gridmet``.
        The files are stored in the NetCDF format and the file names are based
        on the variable names and the years, e.g., ``tmmn_2010.nc``.
    conn_timeout : int, optional
        Connection timeout in seconds, defaults to 1000.
    validate_filesize : bool, optional
        When set to ``True``, the function checks the file size of the previously
        cached files and will re-download if the local filesize does not match
        that of the remote. Defaults to ``True``. Setting this to ``False``
        can be useful when you are sure that the cached files are not corrupted and just
        want to get the combined dataset more quickly. This is faster because it avoids
        web requests that are necessary for getting the file sizes.

    Returns
    -------
    list
        A list of the downloaded files.

    Examples
    --------
    >>> import pygridmet as gridmet
    >>> filenames = gridmet.get_conus(2010, "tmmn")
    """
    yr_list = [years] if isinstance(years, int) else years
    var_list = [variables] if isinstance(variables, str) else variables
    if any(v not in GM_VARS for v in var_list):
        raise InputTypeError("variables", f"one of {GridMET.variables}")
    base_url = "https://www.northwestknowledge.net/metdata/data/{}_{}.nc"
    urls = [base_url.format(v, yr) for v in var_list for yr in yr_list]
    file_names = [Path(save_dir, url.split("/")[-1]) for url in urls]
    return utils.download_files(urls, "nc", file_names, validate_filesize, conn_timeout)
