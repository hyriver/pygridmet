"""Core class for the GridMET functions."""
# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Iterable, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pygridmet.exceptions import InputRangeError, InputTypeError, InputValueError

try:
    from numba import config as numba_config
    from numba import jit, prange

    ngjit = functools.partial(jit, nopython=True, cache=True, nogil=True)
    numba_config.THREADING_LAYER = "workqueue"
    has_numba = True
except ImportError:
    has_numba = False
    prange = range

    T = TypeVar("T")
    Func = Callable[..., T]

    def ngjit(
        signature_or_function: str | Func[T], parallel: bool = False
    ) -> Callable[[Func[T]], Func[T]]:
        def decorator_njit(func: Func[T]) -> Func[T]:
            @functools.wraps(func)
            def wrapper_decorator(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> T:
                return func(*args, **kwargs)

            return wrapper_decorator

        return decorator_njit


if TYPE_CHECKING:
    DF = TypeVar("DF", pd.DataFrame, xr.Dataset)

DATE_FMT = "%Y-%m-%d"
# Default snow params from https://doi.org/10.5194/gmd-11-1077-2018
T_RAIN = 2.5  # degC
T_SNOW = 0.6  # degC

__all__ = ["GridMET"]


@dataclass
class GridMETBase:
    """Base class for validating GridMET requests.

    Parameters
    ----------
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
    variables : list, optional
        List of variables to be downloaded. The acceptable variables are:
        ``pr``, ``rmax``, ``rmin``, ``sph``, ``srad``, ``th``, ``tmmn``, ``tmmx``, ``vs``,
        ``bi``, ``fm100``, ``fm1000``, ``erc``, ``etr``, ``pet``, and ``vpd``.
        Descriptions can be found `here <https://www.climatologylab.org/gridmet.html>`__.
        Defaults to ``None``, i.e., all the variables are downloaded.

    References
    ----------
    .. footbibliography::
    """

    snow: bool
    variables: Iterable[str]

    def __post_init__(self) -> None:
        valid_variables = (
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
        )
        if "all" in self.variables:
            self.variables = valid_variables

        if not set(self.variables).issubset(set(valid_variables)):
            raise InputValueError("variables", valid_variables)

        if self.snow:
            self.variables = list(set(self.variables).union({"tmmn"}))


@ngjit("f8[::1](f8[::1], f8[::1], f8, f8)", parallel=True)
def _separate_snow(
    pr: npt.NDArray[np.float64],
    tmmn: npt.NDArray[np.float64],
    t_rain: np.float64,
    t_snow: np.float64,
) -> npt.NDArray[np.float64]:
    """Separate snow in precipitation."""
    t_rng = t_rain - t_snow
    snow = np.zeros_like(pr)

    for t in prange(pr.shape[0]):
        if tmmn[t] > t_rain:
            snow[t] = 0.0
        elif tmmn[t] < t_snow:
            snow[t] = pr[t]
        else:
            snow[t] = pr[t] * (t_rain - tmmn[t]) / t_rng
    return snow


class GridMET:
    """Base class for GridMET requests.

    Parameters
    ----------
    variables : str or list or tuple, optional
        List of variables to be downloaded. The acceptable variables are:
        ``pr``, ``rmax``, ``rmin``, ``sph``, ``srad``, ``th``, ``tmmn``, ``tmmx``, ``vs``,
        ``bi``, ``fm100``, ``fm1000``, ``erc``, ``etr``, ``pet``, and ``vpd``.
        Descriptions can be found `here <https://www.climatologylab.org/gridmet.html>`__.
        Defaults to ``None``, i.e., all the variables are downloaded.
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        variables: Iterable[str] | str | None = None,
        snow: bool = False,
    ) -> None:
        _variables = ["all"] if variables is None else variables
        _variables = [_variables] if isinstance(_variables, str) else _variables
        validated = GridMETBase(variables=_variables, snow=snow)
        self.variables = validated.variables
        self.snow = validated.snow

        self.bounds = (-124.7666, 25.0666, -67.0583, 49.4000)
        self.valid_start = pd.to_datetime("1980-01-01")
        self.valid_end = datetime.now() - pd.DateOffset(days=1)
        self.missing_value = 32767.0

        self.gridmet_table = pd.DataFrame(
            {
                "variable": [
                    "Precipitation",
                    "Maximum Relative Humidity",
                    "Minimum Relative Humidity",
                    "Specific Humidity",
                    "Surface Radiation",
                    "Wind Direction",
                    "Minimum Air Temperature",
                    "Maximum Air Temperature",
                    "Wind Speed",
                    "Burning Index",
                    "Fuel Moisture (100-hr)",
                    "Fuel Moisture (1000-hr)",
                    "Energy Release Component",
                    "Reference Evapotranspiration (Alfalfa)",
                    "Reference Evapotranspiration (Grass)",
                    "Vapor Pressure Deficit",
                ],
                "abbr": [
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
                ],
                "long_name": [
                    "precipitation_amount",
                    "daily_maximum_relative_humidity",
                    "daily_minimum_relative_humidity",
                    "daily_mean_specific_humidity",
                    "daily_mean_shortwave_radiation_at_surface",
                    "daily_mean_wind_direction",
                    "daily_minimum_temperature",
                    "daily_maximum_temperature",
                    "daily_mean_wind_speed",
                    "daily_mean_burning_index_g",
                    "dead_fuel_moisture_100hr",
                    "dead_fuel_moisture_1000hr",
                    "daily_mean_energy_release_component-g",
                    "daily_mean_reference_evapotranspiration_alfalfa",
                    "daily_mean_reference_evapotranspiration_grass",
                    "daily_mean_vapor_pressure_deficit",
                ],
                "units": [
                    "mm",
                    "%",
                    "%",
                    "kg/kg",
                    "W/m2",
                    "Klockwise from north",
                    "K",
                    "K",
                    "m/s",
                    "-",
                    "%",
                    "%",
                    "-",
                    "mm",
                    "mm",
                    "kPa",
                ],
            }
        )

        self.units = dict(zip(self.gridmet_table["abbr"], self.gridmet_table["units"]))
        self.units["snow"] = "mm"

        self.long_names = dict(zip(self.gridmet_table["abbr"], self.gridmet_table["long_name"]))
        self.long_names["snow"] = "snow_amount"

    @staticmethod
    def check_dates(dates: tuple[str, str] | int | list[int]) -> None:
        """Check if input dates are in correct format and valid."""
        if not isinstance(dates, (tuple, list, int, range)):
            raise InputTypeError(
                "dates",
                "tuple, list, range, or int",
                "(start, end), range(start, end), or [years, ...]",
            )

        if isinstance(dates, tuple) and len(dates) != 2:
            raise InputTypeError("dates", "Start and end should be passed as a tuple of length 2.")

    def dates_todict(self, dates: tuple[str, str]) -> dict[str, str]:
        """Set dates by start and end dates as a tuple, (start, end)."""
        if not isinstance(dates, tuple) or len(dates) != 2:
            raise InputTypeError("dates", "tuple", "(start, end)")

        start = pd.to_datetime(dates[0])
        end = pd.to_datetime(dates[1])

        if start < self.valid_start or end > self.valid_end:
            raise InputRangeError("start/end", f"from {self.valid_start} to {self.valid_end}")

        return {
            "start": start.strftime(DATE_FMT),
            "end": end.strftime(DATE_FMT),
        }

    def years_todict(self, years: list[int] | int | range) -> dict[str, str]:
        """Set date by list of year(s)."""
        years = [years] if isinstance(years, int) else list(years)

        if min(years) < self.valid_start.year or max(years) > self.valid_end.year:
            raise InputRangeError(
                "start/end", f"from {self.valid_start.year} to {self.valid_end.year}"
            )

        return {"years": ",".join(str(y) for y in years)}

    def dates_tolist(self, dates: tuple[str, str]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Correct dates for GridMET accounting for leap years.

        GridMET doesn't account for leap years and removes Dec 31 when
        it's leap year.

        Parameters
        ----------
        dates : tuple
            Target start and end dates.

        Returns
        -------
        list
            All the dates in the GridMET database within the provided date range.
        """
        date_dict = self.dates_todict(dates)
        start = pd.to_datetime(date_dict["start"])
        end = pd.to_datetime(date_dict["end"])

        period = pd.date_range(start, end)
        nl = period[~period.is_leap_year]
        lp = period[(period.is_leap_year) & (~period.strftime(DATE_FMT).str.endswith("12-31"))]
        _period = period[(period.isin(nl)) | (period.isin(lp))]
        years = [_period[_period.year == y] for y in _period.year.unique()]
        return [(y[0], y[-1]) for y in years]

    def years_tolist(self, years: list[int] | int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Correct dates for GridMET accounting for leap years.

        GridMET doesn't account for leap years and removes Dec 31 when
        it's leap year.

        Parameters
        ----------
        years: list
            A list of target years.

        Returns
        -------
        list
            All the dates in the GridMET database within the provided date range.
        """
        date_dict = self.years_todict(years)
        start_list, end_list = [], []
        for year in date_dict["years"].split(","):
            s = pd.to_datetime(f"{year}0101")
            start_list.append(s)
            e = pd.to_datetime(f"{year}1230") if s.is_leap_year else pd.to_datetime(f"{year}1231")
            end_list.append(e)
        return list(zip(start_list, end_list))

    @staticmethod
    def _snow_point(climate: pd.DataFrame, t_rain: float, t_snow: float) -> pd.DataFrame:
        """Separate snow from precipitation."""
        clm = climate.copy()
        clm["snow (mm)"] = _separate_snow(
            clm["pr (mm)"].to_numpy("f8"),
            clm["tmmn (K)"].to_numpy("f8"),
            np.float64(t_rain),
            np.float64(t_snow),
        )
        return clm

    @staticmethod
    def _snow_gridded(climate: xr.Dataset, t_rain: float, t_snow: float) -> xr.Dataset:
        """Separate snow from precipitation."""
        clm = climate.copy()

        def snow_func(
            pr: npt.NDArray[np.float64],
            tmmn: npt.NDArray[np.float64],
            t_rain: float,
            t_snow: float,
        ) -> npt.NDArray[np.float64]:
            """Separate snow based on Martinez and Gupta (2010)."""
            return _separate_snow(
                pr.astype("f8"),
                tmmn.astype("f8") - 273.15,
                np.float64(t_rain),
                np.float64(t_snow),
            )

        clm["snow"] = xr.apply_ufunc(
            snow_func,
            clm["pr"],
            clm["tmmn"] - 273.15,
            t_rain,
            t_snow,
            input_core_dims=[["time"], ["time"], [], []],
            output_core_dims=[["time"]],
            vectorize=True,
            output_dtypes=[clm["pr"].dtype],
        ).transpose("time", "lat", "lon")
        clm["snow"].attrs["units"] = "mm"
        clm["snow"].attrs["long_name"] = "daily snowfall"
        return clm

    def separate_snow(self, clm: DF, t_rain: float = T_RAIN, t_snow: float = T_SNOW) -> DF:
        """Separate snow based on :footcite:t:`Martinez_2010`.

        Parameters
        ----------
        clm : pandas.DataFrame or xarray.Dataset
            Climate data that should include ``pr`` and ``tmmn``.
        t_rain : float, optional
            Threshold for temperature for considering rain, defaults to 2.5 K.
        t_snow : float, optional
            Threshold for temperature for considering snow, defaults to 0.6 K.

        Returns
        -------
        pandas.DataFrame or xarray.Dataset
            Input data with ``snow (mm)`` column if input is a ``pandas.DataFrame``,
            or ``snow`` variable if input is an ``xarray.Dataset``.

        References
        ----------
        .. footbibliography::
        """
        if not has_numba:
            warnings.warn(
                "Numba not installed. Using slow pure python version.", UserWarning, stacklevel=2
            )

        if not isinstance(clm, (pd.DataFrame, xr.Dataset)):
            raise InputTypeError("clm", "pandas.DataFrame or xarray.Dataset")

        if isinstance(clm, xr.Dataset):
            return self._snow_gridded(clm, t_rain, t_snow)
        return self._snow_point(clm, t_rain, t_snow)
