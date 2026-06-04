# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any
import os

from earthkit.data.utils.patterns import Pattern
from earthkit.data import from_source
from multiprocessing.pool import ThreadPool
import numpy as np
import pandas

from ..source import Source
from . import source_registry
from .grib import _expand

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

        self.path = path
        if not select:
            select = "*"
            LOG.warning("No SELECT clause provided; defaulting to all columns.")
        self.select = select
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

    
        # do not substitute if not needed
        if "{" not in self.path:
            paths = [self.path]
        else:
            paths = Pattern(self.path).substitute(date=dates.dates, allow_extra=True)

        result_dfs = []

        # reading large bufr with pdbufr is very slow so we rely on a threadpool to get past the I/O throttle
        max_num_cores = len(os.sched_getaffinity(0))
        num_proc = min(max_num_cores, len(paths))
        if num_proc==1:
            for path in _expand(paths):
                result_dfs.append(self._load_bufr(path))
        else:
            LOG.info(f"Launching BUFR loading threadpool with {num_proc} cores")
            with ThreadPool(num_proc) as p:
                result_dfs = p.map(self._load_bufr,_expand(paths))
        df = pandas.concat(result_dfs)
        LOG.info(f"BUFR source concat {len(df)} rows from {self.path}")
        return df
    
    def _load_bufr(self, path: str):
        self.context.trace("📁", "PATH", path)
        df = bufr_to_df(
            path_str=path,
            select=self.select,
            flavour=self.flavour,
            pivot_columns=self.pivot_columns,
            pivot_values=self.pivot_values,
        )
        LOG.info(f"BUFR source read {len(df)} rows from {self.path}")
        return df

def bufr_to_df(
    path_str: str,
    select: list,
    flavour: dict,
    pivot_columns: list = [],
    pivot_values: list = [],
) -> pandas.DataFrame:

    ds = from_source('file', path_str)

    date_col = flavour["date_column_name"]
    time_col = flavour["time_column_name"]
    lat_col = flavour["latitude_column_name"]
    lon_col = flavour["longitude_column_name"]
    
    select = [date_col,time_col,lat_col,lon_col] + select

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

    # Make sure first 3 columns are time, latitude, longitude
    cols = df.columns.tolist()
    cols.remove("date")
    cols.remove("latitude")
    cols.remove("longitude")
    df = df[["date", "latitude", "longitude"] + cols]

    return df