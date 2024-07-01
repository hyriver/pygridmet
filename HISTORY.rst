=======
History
=======

0.16.1 (2024-XX-XX)
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

0.16.0 (2024-01-03)
-------------------

- Initial release on PyPI.
