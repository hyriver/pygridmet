=======
History
=======

0.19.1 (2025-01-17)
-------------------

This release is a major refactoring of the package to make it more lightweight
and independent. The package now uses ``aiohttp`` and ``aiofiles`` with a limit
on the number of simulnatious connections to the host, for handling
all server requests. This avoids hammering the server with multiple requests and
improves the performance and reliability of the package. The package no longer
depends on other HyRiver libraries, making it more lightweight and faster to load.

Internal Changes
~~~~~~~~~~~~~~~~
- Remove dependency on other HyRiver libraries to make the package more
  lightweight and independent.
- Use ``aiohttp`` and ``aiofiles`` for handling all server requests.
  This avoids hammering the server with multiple requests and improves
  the performance and reliability of the package.
- Remove dependency on ``geopandas`` and use ``shapely`` only for handling
  geometries. This makes the package more lightweight and faster.

Breaking Changes
~~~~~~~~~~~~~~~~
- Remove the option to disable SSL in all functions. Now, SSL verification
  is always enabled.

0.18.0 (2024-10-05)
-------------------

New Features
~~~~~~~~~~~~
- Add a new function called ``get_conus`` for downloading the daily
  meteorological data for the entire contiguous United States (CONUS) from
  the GridMet service, for given years and variables. This can be accessed
  from the CLI using, for example, ``pygridmet conus -y 2010 -v tmmn`` command.

Breaking Changes
~~~~~~~~~~~~~~~~
- Drop support for Python 3.8 since its end-of-life date is October 2024.
- Remove all exceptions from the main module and raise them from the
  ``exceptions`` module. This is to declutter the public API and make
  it easier to maintain.

0.17.1 (2024-09-14)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- A release without change to match the version of other HyRiver packages.

0.17.0 (2024-05-07)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Add the ``exceptions`` module to the high-level API to declutter
  the main module. In the future, all exceptions will be raised from
  this module and not from the main module. For now, the exceptions
  are raised from both modules for backward compatibility.
- Switch to using the ``src`` layout instead of the ``flat`` layout
  for the package structure. This is to make the package more
  maintainable and to avoid any potential conflicts with other
  packages.
- Add artifact attestations to the release workflow.

0.16.0 (2024-01-03)
-------------------

- Initial release on PyPI.
