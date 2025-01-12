"""Some utilities for PyDaymet."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import numpy as np
import pyproj
import shapely
import urllib3
from pyproj import Transformer
from pyproj.exceptions import CRSError as ProjCRSError
from rioxarray.exceptions import OneDimensionalRaster
from shapely import MultiPolygon, Polygon, ops

from pygridmet.exceptions import DownloadError, InputRangeError, InputTypeError

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray
    from shapely.geometry.base import BaseGeometry

    CRSTYPE = int | str | pyproj.CRS
    POLYTYPE = Polygon | MultiPolygon | tuple[float, float, float, float]
    NUMBER = int | float | np.number

__all__ = [
    "clip_dataset",
    "download_files",
    "to_geometry",
    "transform_coords",
    "validate_coords",
    "validate_crs",
    "write_crs",
]
CHUNK_SIZE = 1048576  # 1 MB
TransformerFromCRS = lru_cache(Transformer.from_crs)


def validate_crs(crs: CRSTYPE) -> str:
    """Validate a CRS.

    Parameters
    ----------
    crs : str, int, or pyproj.CRS
        Input CRS.

    Returns
    -------
    str
        Validated CRS as a string.
    """
    try:
        return pyproj.CRS(crs).to_string()
    except ProjCRSError as ex:
        raise InputTypeError("crs", "a valid CRS") from ex


def transform_coords(
    coords: Sequence[tuple[float, float]], in_crs: CRSTYPE, out_crs: CRSTYPE
) -> list[tuple[float, float]]:
    """Transform coordinates from one CRS to another."""
    try:
        pts = shapely.points(coords)
    except (TypeError, AttributeError, ValueError) as ex:
        raise InputTypeError("coords", "a list of tuples") from ex
    x, y = shapely.get_coordinates(pts).T
    x_proj, y_proj = TransformerFromCRS(in_crs, out_crs, always_xy=True).transform(x, y)
    return list(zip(x_proj, y_proj))


def _geo_transform(geom: BaseGeometry, in_crs: CRSTYPE, out_crs: CRSTYPE) -> BaseGeometry:
    """Transform a geometry from one CRS to another."""
    project = TransformerFromCRS(in_crs, out_crs, always_xy=True).transform
    return ops.transform(project, geom)


def validate_coords(
    coords: Iterable[tuple[float, float]], bounds: tuple[float, float, float, float]
) -> NDArray[np.float64]:
    """Validate coordinates within a bounding box."""
    try:
        pts = shapely.points(list(coords))
    except (TypeError, AttributeError, ValueError) as ex:
        raise InputTypeError("coords", "a list of tuples") from ex
    if shapely.contains(shapely.box(*bounds), pts).all():
        return shapely.get_coordinates(pts)
    raise InputRangeError("coords", f"within {bounds}")


def to_geometry(
    geometry: BaseGeometry | tuple[float, float, float, float],
    geo_crs: CRSTYPE | None = None,
    crs: CRSTYPE | None = None,
) -> BaseGeometry:
    """Return a Shapely geometry and optionally transformed to a new CRS.

    Parameters
    ----------
    geometry : shaple.Geometry or tuple of length 4
        Any shapely geometry object or a bounding box (minx, miny, maxx, maxy).
    geo_crs : int, str, or pyproj.CRS, optional
        Spatial reference of the input geometry, defaults to ``None``.
    crs : int, str, or pyproj.CRS
        Target spatial reference, defaults to ``None``.

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        A shapely geometry object.
    """
    is_geom = np.atleast_1d(shapely.is_geometry(geometry))
    if is_geom.all() and len(is_geom) == 1:
        geom = geometry
    elif isinstance(geometry, Iterable) and len(geometry) == 4 and np.isfinite(geometry).all():
        geom = shapely.box(*geometry)
    else:
        raise InputTypeError("geometry", "a shapley geometry or tuple of length 4")

    if geo_crs is not None and crs is not None:
        return _geo_transform(geom, geo_crs, crs)
    elif geo_crs is None and crs is not None:
        return geom
    raise InputTypeError("geo_crs/crs", "either both None or both valid CRS")


def _download(url: str, fname: Path, http: urllib3.HTTPConnectionPool) -> None:
    """Download a file from a URL."""
    parsed_url = urlparse(url)
    path = f"{parsed_url.path}?{parsed_url.query}"
    head = http.request("HEAD", path)
    fsize = int(head.headers.get("Content-Length", -1))
    if fname.exists() and fname.stat().st_size == fsize:
        return
    fname.unlink(missing_ok=True)
    fname.write_bytes(http.request("GET", path).data)


def download_files(url_list: list[str], file_extension: str, save_dir: Path | None = None, rewrite: bool = False) -> list[Path]:
    """Download multiple files concurrently."""
    if save_dir is None:
        hr_cache = os.getenv("HYRIVER_CACHE_NAME")
        save_dir = Path(hr_cache).parent if hr_cache else Path("cache")
    save_dir.mkdir(exist_ok=True, parents=True)

    http = urllib3.HTTPConnectionPool(
        urlparse(url_list[0]).netloc,
        maxsize=10,
        block=True,
        retries=urllib3.Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 504],
            allowed_methods=["HEAD", "GET"],
        ),
    )

    file_list = [
        Path(save_dir, f"{hashlib.sha256(url.encode()).hexdigest()}.{file_extension}")
        for url in url_list
    ]
    if rewrite:
        _ = [f.unlink(missing_ok=True) for f in file_list]
    max_workers = min(4, os.cpu_count() or 1, len(url_list))
    if max_workers == 1:
        _ = [_download(url, path, http) for url, path in zip(url_list, file_list)]
        return file_list

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(_download, url, path, http): url
            for url, path in zip(url_list, file_list)
        }
        for future in as_completed(future_to_url):
            try:
                future.result()
            except Exception as e:  # noqa: PERF203
                raise DownloadError(future_to_url[future], e) from e
    return file_list


def clip_dataset(
    ds: xr.Dataset,
    geometry: Polygon | MultiPolygon | tuple[float, float, float, float],
    crs: CRSTYPE,
) -> xr.Dataset:
    """Mask a ``xarray.Dataset`` based on a geometry."""
    attrs = {v: ds[v].attrs for v in ds}

    geom = to_geometry(geometry, crs, ds.rio.crs)
    try:
        ds = ds.rio.clip_box(*geom.bounds, auto_expand=True)
        if isinstance(geometry, (Polygon, MultiPolygon)):
            ds = ds.rio.clip([geom])
    except OneDimensionalRaster:
        ds = ds.rio.clip([geom], all_touched=True)

    _ = [ds[v].rio.update_attrs(attrs[v], inplace=True) for v in ds]
    ds.rio.update_encoding(ds.encoding, inplace=True)
    return ds
