.. image:: https://raw.githubusercontent.com/hyriver/HyRiver-examples/main/notebooks/_static/pygridmet_logo.png
    :target: https://github.com/hyriver/HyRiver

|

.. image:: https://joss.theoj.org/papers/b0df2f6192f0a18b9e622a3edff52e77/status.svg
    :target: https://joss.theoj.org/papers/b0df2f6192f0a18b9e622a3edff52e77
    :alt: JOSS

|

.. |pygeohydro| image:: https://github.com/hyriver/pygeohydro/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pygeohydro/actions/workflows/test.yml
    :alt: Github Actions

.. |pygeoogc| image:: https://github.com/hyriver/pygeoogc/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pygeoogc/actions/workflows/test.yml
    :alt: Github Actions

.. |pygeoutils| image:: https://github.com/hyriver/pygeoutils/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pygeoutils/actions/workflows/test.yml
    :alt: Github Actions

.. |pynhd| image:: https://github.com/hyriver/pynhd/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pynhd/actions/workflows/test.yml
    :alt: Github Actions

.. |py3dep| image:: https://github.com/hyriver/py3dep/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/py3dep/actions/workflows/test.yml
    :alt: Github Actions

.. |pydaymet| image:: https://github.com/hyriver/pydaymet/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pydaymet/actions/workflows/test.yml
    :alt: Github Actions

.. |pygridmet| image:: https://github.com/hyriver/pygridmet/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pygridmet/actions/workflows/test.yml
    :alt: Github Actions

.. |pynldas2| image:: https://github.com/hyriver/pynldas2/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pynldas2/actions/workflows/test.yml
    :alt: Github Actions

.. |async| image:: https://github.com/hyriver/async-retriever/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/async-retriever/actions/workflows/test.yml
    :alt: Github Actions

.. |signatures| image:: https://github.com/hyriver/hydrosignatures/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/hydrosignatures/actions/workflows/test.yml
    :alt: Github Actions

================ ==================================================================== ============
Package          Description                                                          Status
================ ==================================================================== ============
PyNHD_           Navigate and subset NHDPlus (MR and HR) using web services           |pynhd|
Py3DEP_          Access topographic data through National Map's 3DEP web service      |py3dep|
PyGeoHydro_      Access NWIS, NID, WQP, eHydro, NLCD, CAMELS, and SSEBop databases    |pygeohydro|
PyDaymet_        Access daily, monthly, and annual climate data via Daymet            |pydaymet|
PyGridMET_       Access daily climate data via GridMET                                |pygridmet|
PyNLDAS2_        Access hourly NLDAS-2 data via web services                          |pynldas2|
HydroSignatures_ A collection of tools for computing hydrological signatures          |signatures|
AsyncRetriever_  High-level API for asynchronous requests with persistent caching     |async|
PyGeoOGC_        Send queries to any ArcGIS RESTful-, WMS-, and WFS-based services    |pygeoogc|
PyGeoUtils_      Utilities for manipulating geospatial, (Geo)JSON, and (Geo)TIFF data |pygeoutils|
================ ==================================================================== ============

.. _PyGeoHydro: https://github.com/hyriver/pygeohydro
.. _AsyncRetriever: https://github.com/hyriver/async-retriever
.. _PyGeoOGC: https://github.com/hyriver/pygeoogc
.. _PyGeoUtils: https://github.com/hyriver/pygeoutils
.. _PyNHD: https://github.com/hyriver/pynhd
.. _Py3DEP: https://github.com/hyriver/py3dep
.. _PyDaymet: https://github.com/hyriver/pydaymet
.. _PyGridMET: https://github.com/hyriver/pygridmet
.. _PyNLDAS2: https://github.com/hyriver/pynldas2
.. _HydroSignatures: https://github.com/hyriver/hydrosignatures

PyGridMET: Daily climate data through GridMET
---------------------------------------------

.. image:: https://img.shields.io/pypi/v/pygridmet.svg
    :target: https://pypi.python.org/pypi/pygridmet
    :alt: PyPi

.. image:: https://img.shields.io/conda/vn/conda-forge/pygridmet.svg
    :target: https://anaconda.org/conda-forge/pygridmet
    :alt: Conda Version

.. image:: https://codecov.io/gh/hyriver/pygridmet/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/hyriver/pygridmet
    :alt: CodeCov

.. image:: https://img.shields.io/pypi/pyversions/pygridmet.svg
    :target: https://pypi.python.org/pypi/pygridmet
    :alt: Python Versions

.. image:: https://static.pepy.tech/badge/pygridmet
    :target: https://pepy.tech/project/pygridmet
    :alt: Downloads

|

.. image:: https://www.codefactor.io/repository/github/hyriver/pygridmet/badge
   :target: https://www.codefactor.io/repository/github/hyriver/pygridmet
   :alt: CodeFactor

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://github.com/pre-commit/pre-commit
    :alt: pre-commit

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/hyriver/HyRiver-examples/main?urlpath=lab/tree/notebooks
    :alt: Binder

|

Features
--------

PyGridMET is a part of `HyRiver <https://github.com/hyriver/HyRiver>`__ software stack that
is designed to aid in hydroclimate analysis through web services. This package provides
access to daily climate data over contermonious US (CONUS) from
`GridMET <https://www.climatologylab.org/gridmet.html>`__ database using NetCDF
Subset Service (NCSS). Both single pixel (using ``get_bycoords`` function) and gridded data (using
``get_bygeom``) are supported which are returned as
``pandas.DataFrame`` and ``xarray.Dataset``, respectively.

You can find some example notebooks `here <https://github.com/hyriver/HyRiver-examples>`__.

Moreover, under the hood, PyGridMET uses
`PyGeoOGC <https://github.com/hyriver/pygeoogc>`__ and
`AsyncRetriever <https://github.com/hyriver/async-retriever>`__ packages
for making requests in parallel and storing responses in chunks. This improves the
reliability and speed of data retrieval significantly.

You can control the request/response caching behavior and verbosity of the package
by setting the following environment variables:

* ``HYRIVER_CACHE_NAME``: Path to the caching SQLite database for asynchronous HTTP
  requests. It defaults to ``./cache/aiohttp_cache.sqlite``
* ``HYRIVER_CACHE_NAME_HTTP``: Path to the caching SQLite database for HTTP requests.
  It defaults to ``./cache/http_cache.sqlite``
* ``HYRIVER_CACHE_EXPIRE``: Expiration time for cached requests in seconds. It defaults to
  one week.
* ``HYRIVER_CACHE_DISABLE``: Disable reading/writing from/to the cache. The default is false.
* ``HYRIVER_SSL_CERT``: Path to a SSL certificate file.

For example, in your code before making any requests you can do:

.. code-block:: python

    import os

    os.environ["HYRIVER_CACHE_NAME"] = "path/to/aiohttp_cache.sqlite"
    os.environ["HYRIVER_CACHE_NAME_HTTP"] = "path/to/http_cache.sqlite"
    os.environ["HYRIVER_CACHE_EXPIRE"] = "3600"
    os.environ["HYRIVER_CACHE_DISABLE"] = "true"
    os.environ["HYRIVER_SSL_CERT"] = "path/to/cert.pem"

You can also try using PyGridMET without installing
it on your system by clicking on the binder badge. A Jupyter Lab
instance with the HyRiver stack pre-installed will be launched in your web browser, and you
can start coding!

Moreover, requests for additional functionalities can be submitted via
`issue tracker <https://github.com/hyriver/pygridmet/issues>`__.

Citation
--------
If you use any of HyRiver packages in your research, we appreciate citations:

.. code-block:: bibtex

    @article{Chegini_2021,
        author = {Chegini, Taher and Li, Hong-Yi and Leung, L. Ruby},
        doi = {10.21105/joss.03175},
        journal = {Journal of Open Source Software},
        month = {10},
        number = {66},
        pages = {1--3},
        title = {{HyRiver: Hydroclimate Data Retriever}},
        volume = {6},
        year = {2021}
    }

Installation
------------

You can install PyGridMET using ``pip`` as follows:

.. code-block:: console

    $ pip install pygridmet

Alternatively, PyGridMET can be installed from the ``conda-forge`` repository
using `Conda <https://docs.conda.io/en/latest/>`__:

.. code-block:: console

    $ conda install -c conda-forge pygridmet

Quick start
-----------

You can use PyGridMET using command-line or as a Python library. The commanda-line
provides access to two functionality:

- Getting gridded climate data: You must create a ``geopandas.GeoDataFrame`` that contains
  the geometries of the target locations. This dataframe must have four columns:
  ``id``, ``start``, ``end``, ``geometry``. The ``id`` column is used as
  filenames for saving the obtained climate data to a NetCDF (``.nc``) file. The ``start``
  and ``end`` columns are starting and ending dates of the target period. Then,
  you must save the dataframe as a shapefile (``.shp``) or geopackage (``.gpkg``) with
  CRS attribute.
- Getting single pixel climate data: You must create a CSV file that
  contains coordinates of the target locations. This file must have at four columns:
  ``id``, ``start``, ``end``, ``lon``, and ``lat``. The ``id`` column is used as filenames
  for saving the obtained climate data to a CSV (``.csv``) file. The ``start`` and ``end``
  columns are the same as the ``geometry`` command. The ``lon`` and ``lat`` columns are
  the longitude and latitude coordinates of the target locations.

.. code-block:: console

    $ pygridmet -h
    Usage: pygridmet [OPTIONS] COMMAND [ARGS]...

    Command-line interface for PyGridMET.

    Options:
    -h, --help  Show this message and exit.

    Commands:
    coords    Retrieve climate data for a list of coordinates.
    geometry  Retrieve climate data for a dataframe of geometries.

The ``coords`` sub-command is as follows:

.. code-block:: console

    $ pygridmet coords -h
    Usage: pygridmet coords [OPTIONS] FPATH

    Retrieve climate data for a list of coordinates.

    FPATH: Path to a csv file with four columns:
        - ``id``: Feature identifiers that gridmet uses as the output netcdf filenames.
        - ``start``: Start time.
        - ``end``: End time.
        - ``lon``: Longitude of the points of interest.
        - ``lat``: Latitude of the points of interest.
        - ``snow``: (optional) Separate snowfall from precipitation, default is ``False``.

    Examples:
        $ cat coords.csv
        id,lon,lat,start,end
        california,-122.2493328,37.8122894,2012-01-01,2014-12-31
        $ pygridmet coords coords.csv -v pr -v tmmn

    Options:
    -v, --variables TEXT  Target variables. You can pass this flag multiple
                            times for multiple variables.
    -s, --save_dir PATH   Path to a directory to save the requested files.
                            Extension for the outputs is .nc for geometry and .csv
                            for coords.
    --disable_ssl         Pass to disable SSL certification verification.
    -h, --help            Show this message and exit.

And, the ``geometry`` sub-command is as follows:

.. code-block:: console

    $ pygridmet geometry -h
    Usage: pygridmet geometry [OPTIONS] FPATH

    Retrieve climate data for a dataframe of geometries.

    FPATH: Path to a shapefile (.shp) or geopackage (.gpkg) file.
    This file must have four columns and contain a ``crs`` attribute:
        - ``id``: Feature identifiers that gridmet uses as the output netcdf filenames.
        - ``start``: Start time.
        - ``end``: End time.
        - ``geometry``: Target geometries.
        - ``snow``: (optional) Separate snowfall from precipitation, default is ``False``.

    Examples:
        $ pygridmet geometry geo.gpkg -v pr -v tmmn

    Options:
    -v, --variables TEXT  Target variables. You can pass this flag multiple
                            times for multiple variables.
    -s, --save_dir PATH   Path to a directory to save the requested files.
                            Extension for the outputs is .nc for geometry and .csv
                            for coords.
    --disable_ssl         Pass to disable SSL certification verification.
    -h, --help            Show this message and exit.

Now, let's see how we can use PyGridMET as a library.

PyGridMET offers two functions for getting climate data; ``get_bycoords`` and ``get_bygeom``.
The arguments of these functions are identical except the first argument where the latter
should be polygon and the former should be a coordinate (a tuple of length two as in (x, y)).
The input geometry or coordinate can be in any valid CRS (defaults to ``EPSG:4326``). The
``dates`` argument can be either a tuple of length two like ``(start_str, end_str)`` or a list of
years like ``[2000, 2005]``. It is noted that both functions have a ``snow`` flag for separating
snow from precipitation using
`Martinez and Gupta (2010) <https://doi.org/10.1029/2009WR008294>`__ method.

We can get a dataframe of available variables and their info by calling
``GridMET().gridmet_table``:

+----------------------------------------+------------+------------------------------+
| Variable                               | Abbr       | Unit                         |
+========================================+============+==============================+
| Precipitation                          | ``pr``     | mm                           |
+----------------------------------------+------------+------------------------------+
| Maximum Relative Humidity              | ``rmax``   | %                            |
+----------------------------------------+------------+------------------------------+
| Minimum Relative Humidity              | ``rmin``   | %                            |
+----------------------------------------+------------+------------------------------+
| Specific Humidity                      | ``sph``    | kg/kg                        |
+----------------------------------------+------------+------------------------------+
| Surface Radiation                      | ``srad``   | W/m2                         |
+----------------------------------------+------------+------------------------------+
| Wind Direction                         | ``th``     | Degrees Clockwise from north |
+----------------------------------------+------------+------------------------------+
| Minimum Air Temperature                | ``tmmn``   | K                            |
+----------------------------------------+------------+------------------------------+
| Maximum Air Temperature                | ``tmmx``   | K                            |
+----------------------------------------+------------+------------------------------+
| Wind Speed                             | ``vs``     | m/s                          |
+----------------------------------------+------------+------------------------------+
| Burning Index                          | ``bi``     | Dimensionless                |
+----------------------------------------+------------+------------------------------+
| Fuel Moisture (100-hr)                 | ``fm100``  | %                            |
+----------------------------------------+------------+------------------------------+
| Fuel Moisture (1000-hr)                | ``fm1000`` | %                            |
+----------------------------------------+------------+------------------------------+
| Energy Release Component               | ``erc``    | Dimensionless                |
+----------------------------------------+------------+------------------------------+
| Reference Evapotranspiration (Alfalfa) | ``etr``    | mm                           |
+----------------------------------------+------------+------------------------------+
| Reference Evapotranspiration (Grass)   | ``pet``    | mm                           |
+----------------------------------------+------------+------------------------------+
| Vapor Pressure Deficit                 | ``vpd``    | kPa                          |
+----------------------------------------+------------+------------------------------+

.. code-block:: python

    from pynhd import NLDI
    import pygridmet as gridmet

    geometry = NLDI().get_basins("01031500").geometry[0]

    var = ["pr", "tmmn"]
    dates = ("2000-01-01", "2000-06-30")

    daily = gridmet.get_bygeom(geometry, dates, variables=var, snow=True)

.. image:: https://raw.githubusercontent.com/hyriver/HyRiver-examples/main/notebooks/_static/gridmet_grid.png
    :target: https://github.com/hyriver/HyRiver-examples/blob/main/notebooks/gridmet.ipynb

If the input geometry (or coordinate) is in a CRS other than ``EPSG:4326``, we should pass
it to the functions.

.. code-block:: python

    coords = (-1431147.7928, 318483.4618)
    crs = 3542
    dates = ("2000-01-01", "2006-12-31")
    data = gridmet.get_bycoords(coords, dates, variables=var, loc_crs=crs)

.. image:: https://raw.githubusercontent.com/hyriver/HyRiver-examples/main/notebooks/_static/gridmet_loc.png
    :target: https://github.com/hyriver/HyRiver-examples/blob/main/notebooks/gridmet.ipynb

Additionally, the ``get_bycoords`` function accepts a list of coordinates and by setting the
``to_xarray`` flag to ``True`` it can return the results as a ``xarray.Dataset`` instead of
a ``pandas.DataFrame``:

.. code-block:: python

    coords = [(-94.986, 29.973), (-95.478, 30.134)]
    idx = ["P1", "P2"]
    clm_ds = gridmet.get_bycoords(coords, range(2000, 2021), coords_id=idx, to_xarray=True)

Contributing
------------

Contributions are very welcomed. Please read
`CONTRIBUTING.rst <https://github.com/hyriver/pygridmet/blob/main/CONTRIBUTING.rst>`__
file for instructions.
