# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import earthkit.data as ekd
import numpy as np
import pandas
import yaml

from ..source import Source
from . import source_registry

LOG = logging.getLogger(__name__)


@source_registry.register("bufr")
class BufrSource(Source):
    """BUFR data source."""

    emoji = "📥"

    def __init__(
        self,
        context,
        path: str,
        select: str | None = None,
        whitelist: dict | None = None,
        flavour: dict = {},
        pivot_columns: list = [],
        pivot_values: list = [],
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialise the BUFR input.

        Parameters
        ----------
        context : dict
            The context.
        path : str
            The path to the BUFR file.
        select : str, optional
            The select clause. Defaults to all columns ("*").
        whitelist : str, optional
            The whitelist function (equiv to where clause). Defaults to no additional filtering ("").
        flavour : dict, optional
            Naming of the latitude, longitude, date and time columns. Defaults to
            {"latitude_column_name": "lat",
            "longitude_column_name": "lon",
            "date_column_name": "date",
            "time_column_name": "time"}.
        pivot_columns : list, optional
            List of column names - values in these columns will be used to
            define the new columns after the reshaping.
            Typically these identify entries in `pivot_values` as belonging to
            a particular observation type: for instance "channel_number" or
            "varno". Defaults to [].
        pivot_values : list, optional
            List of column names - values in these columns will
            be spread across different values of the columns. For instance,
            "observed_value" and "quality_control_value". Defaults to [].
        kwargs : dict, optional
            Additional keyword arguments.

        Note: All columns not specified in "pivot_columns" and "pivot_values"
            will be assumed to be "index" values (i.e. are the same within a
            given observation group).

        Note: Pivot values are named according to the unique values in the
            pivot columns. For instance, if
            `pivot_columns=["channel_number@body"]`
            with two unique channel numbers 1 and 2 that identify rows, and
            `pivot_values=["initial_obsvalue@body"]`, then the resulting columns
            will be named "observed_value_1" and "observed_value_2".

        """
        super().__init__(context)

        self.path = path
        if not select:
            select = "*"
            LOG.warning("No SELECT clause provided; defaulting to all columns.")
        if not whitelist:
            LOG.warning("No whitelist provided; defaulting to no additional filtering.")
        self.select = select
        self.whitelist = whitelist
        self.flavour = {
            "latitude_column_name": "lat@hdr",
            "longitude_column_name": "lon@hdr",
            "date_column_name": "date@hdr",
            "time_column_name": "time@hdr",
        }
        self.flavour.update(flavour)
        self.pivot_columns = pivot_columns
        self.pivot_values = pivot_values

    def execute(self, dates: Any) -> pandas.DataFrame:
        """Execute the BUFR source.

        Parameters
        ----------
        dates : Any
            The input dates.

        Returns
        -------
        pandas.dataframe.DataFrame
            The output dataframe.
        """
        start = np.datetime_as_string(dates.start_range)
        end = np.datetime_as_string(dates.end_range)

        df = bufr_to_df(
            start=start,
            end=end,
            path_str=self.path,
            select=self.select,
            whitelist=self.whitelist,
            flavour=self.flavour,
            pivot_columns=self.pivot_columns,
            pivot_values=self.pivot_values,
        )
        LOG.info(f"BUFR source read {len(df)} rows from {self.path}")
        LOG.info(df.head())
        return df

def bufr_to_df(
    start: np.datetime64,
    end: np.datetime64,
    path_str: str,
    select: list,
    whitelist: dict,
    flavour: dict,
    pivot_columns: list = [],
    pivot_values: list = [],
) -> pandas.DataFrame:

    ds = ekd.from_source('file', path_str)

    date_col = flavour["date_column_name"]
    time_col = flavour["time_column_name"]
    lat_col = flavour["latitude_column_name"]
    lon_col = flavour["longitude_column_name"]
    
    if whitelist is not None:
        whitedict = get_whitelist_filter(whitelist, start, end)
        wl_keys = list(whitedict.keys())
    else:
        wl_keys = []
        whitedict = dict()
    select = [date_col,time_col,lat_col,lon_col] + select + wl_keys

    df = ds.to_pandas(
    columns=tuple(select)
    )
    
    # The new "datetime" column has to be constructed from the existing date and
    # time columns which have them as YYYYMMDD and HHMMSS integers
    df["date"] = pandas.to_datetime(
        df[date_col].astype(str).str.zfill(8)
        + df[time_col].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
    )
    df.drop(columns=[flavour["date_column_name"], flavour["time_column_name"]], inplace=True)
    
     # The latitude and longitude columns may need renaming to standard names
    if flavour["latitude_column_name"] != "latitude":
        df.rename(
            columns={flavour["latitude_column_name"]: "latitude"},
            inplace=True,
        )
    if flavour["longitude_column_name"] != "longitude":
        df.rename(
            columns={flavour["longitude_column_name"]: "longitude"},
            inplace=True,
        )

    # filter data from whitelist
    
    for key, val in whitedict.items():
        df = df.loc[df[key].isin(val)]
    df = df.drop(columns=wl_keys)

    # Make sure first 3 columns are time, latitude, longitude
    cols = df.columns.tolist()
    cols.remove("date")
    cols.remove("latitude")
    cols.remove("longitude")
    df = df[["date", "latitude", "longitude"] + cols]

    return df

def get_whitelist_filter(whitelist: dict, start: np.datetime64, end: np.datetime64) -> tuple[list, list]:
    """Create the keys and values to filter dataframe from config 
    Special case is whitelist is taken from file : refine whitelist with date-dependent filter
    """

    filter_format = whitelist.get('format', 'from_lists')

    match filter_format:
        case 'from_lists':
            whitedict = whitelist.get('lists', None)

            assert whitedict is not None, ("Whitelist format is from_lists"
                                      "(default), got no information")
        case 'from_file':
            filename = whitelist.get('path')
            with open(filename,'r') as f:
                whitelist_data = yaml.safe_load(f)

            start_dt, end_dt = iso8601_to_datetime(start), iso8601_to_datetime(end)
            periods = whitelist_data.keys()
            found_period = False
            for period in periods:
                period_start, period_end = period.split('/')
                period_start, period_end = iso8601_to_datetime(period_start), iso8601_to_datetime(period_end)
                if start_dt >= period_start and end_dt <= period_end:
                    whitedict = whitelist_data.get(period,None)

                    assert whitedict is not None, ("Whitelist format is from_file,"
                                      f"but got no dict for {period}")
                    found_period = True
                    break
            if not found_period:
                raise ValueError(f"No period found for whitelist {filename}, start {start}, end {end}")
        
        case _:
            raise ValueError("Unknown whitelist format")
        
    return whitedict

def iso8601_to_datetime(iso8601_str: str) -> str:
    """Convert ISO8601 datetime string to YYYYMMDDHHMMSS string.

    Parameters
    ----------
    iso8601_str : str
        ISO8601 datetime string.

    Returns
    -------
    str
        Datetime string in YYYYMMDDHHMMSS format.
    """
    dt = datetime.fromisoformat(iso8601_str)
    return dt