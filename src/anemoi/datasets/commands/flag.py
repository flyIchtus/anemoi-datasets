# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import os
import uuid
from typing import Any

from anemoi.datasets.create.dataset import Dataset
from anemoi.datasets.create.recipe import Dates, Recipe, loader_recipe_from_yaml

from . import Command

LOG = logging.getLogger(__name__)


class Flag(Command):
    """Flag dates in a dataset.

    Assign a new UUID, update the dataset name stored in the recipe metadata,
    and move the store to the new path.
    """

    def add_arguments(self, parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        parser : Any
            The command parser to which arguments are added.
        """
        parser.add_argument("recipe", help="Path of the recipe with flagged dates.")
        parser.add_argument("target", help="Path for the dataset (must end in .zarr).")

    def run(self, args: Any) -> None:
        """Execute the rename command.

        Parameters
        ----------
        args : Any
            The arguments passed to the command.
        """
        recipe = args.source.rstrip("/")
        target = args.target.rstrip("/")

        # --- validation ---
        if not target.endswith(".zarr"):
            raise ValueError("Target must end in '.zarr'.")
        if not os.path.exists(recipe):
            raise FileNotFoundError(f"Source recipe '{recipe}' does not exist.")
        if not os.path.exists(target):
            raise  FileNotFoundError(f"Target dataset '{target}' does not exist.")

        new_recipe = loader_recipe_from_yaml(recipe)

        # --- update metadata in place on the source store ---
        dataset = Dataset(target, update=True)

        new_uuid = str(uuid.uuid4())
        old_uuid = dataset.get_metadata("uuid")

        # Update the name inside the recipe(s).
        for key in ("recipe", "_recipe"):
            old_recipe = dataset.get_metadata(key)
            if old_recipe is None:
                continue
            old_recipe = recipe if isinstance(recipe, dict) else json.loads(recipe)
            old_recipe = Recipe(**old_recipe)

        # Check that old recipe and new recipe match except for flagged dates
        for key, value in old_recipe.items():
            if key!="flagged" and new_recipe[key] != value:
                raise ValueError(f'Original dataset recipe mismatches current on {key}')
            if key=="flagged" and new_recipe[key]==value:
                LOG.warning("Identical flagged dates in dataset and recipe: no-op")
                return
            if key=="flagged" and new_recipe[key] is not None:
                LOG.warning("⚠️ Overwriting previously flagged dates")

        flagged = {k : [_.isoformat() for _ in v] for k,v in new_recipe.dates.flagged}
        filtered_flagged = {}
        dates = new_recipe.dates.full_dates_list()
        for k in flagged:
            correct_flagged = [v for v in flagged[k] if v in dates]
            filtered_flagged[k] = correct_flagged

        dataset.update_metadata({"flagged_dates": filtered_flagged})
        
        # Track the chain of previous UUIDs (for provenance only).
        previous_uuids = dataset.get_metadata("previous_uuids", [])
        if old_uuid is not None:
            previous_uuids.append(old_uuid)

        dataset.update_metadata(uuid=new_uuid, previous_uuids=previous_uuids)

        LOG.info(f"Flagged  from  recipe '{recipe}' -> '{target}': uuid {old_uuid} -> {new_uuid}")


    @staticmethod
    def _add_flagged_dates_in_recipe(recipe: Any, flagged: Dates) -> Any:
        """Set the top-level ``name`` field of the recipe to *new_name*.

        Parameters
        ----------
        recipe : Any
            The recipe metadata, either a dict (the sanitised ``recipe`` attr)
            or a JSON string (the ``_recipe`` attr).
        new_name : str
            The new dataset name to store in the recipe.

        Returns
        -------
        Any
            The recipe with its ``name`` field updated, in the same form
            (dict or JSON string) as the input.
        """
        is_str = isinstance(recipe, str)
        data = json.loads(recipe) if is_str else dict(recipe)
        data['flagged'] = flagged
        return json.dumps(data) if is_str else data


command = Flag
