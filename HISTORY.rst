=======
History
=======

0.17.1 (2024-09-14)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Drop support for Python 3.8 since its end-of-life date is October 2024.

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
