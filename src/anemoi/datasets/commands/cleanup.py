# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import time
from typing import Any

from anemoi.utils.humanize import seconds_to_human

from anemoi.datasets.commands.create import task

from . import Command

LOG = logging.getLogger(__name__)


class Cleanup(Command):
    """Create a dataset, step by step."""

    internal = True
    timestamp = True

    def add_arguments(self, subparser: Any) -> None:
        """Add command line arguments to the parser.

        Parameters
        ----------
        subparser : Any
            The argument parser.
        """
        subparser.add_argument("path", help="Path to store the created data.")
        subparser.add_argument(
            "--delta",
            help="Compute statistics tendencies on a given time delta, if possible. Must be a multiple of the frequency.",
            nargs="+",
        )

    def run(self, args: Any) -> None:
        """Execute the cleanup command.

        Parameters
        ----------
        args : Any
            The command line arguments.
        """
        options = vars(args)
        options.pop("command")
        now = time.time()
        step = self.__class__.__name__.lower()

        if "version" in options:
            options.pop("version")

        if "debug" in options:
            options.pop("debug")

        task(step, options)

        LOG.info(f"Create step '{step}' completed in {seconds_to_human(time.time()-now)}")


command = Cleanup
