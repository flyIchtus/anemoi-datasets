.. _using_flagged_dates:

######################
 Flagged dates
######################

*Anemoi* allows to flag some dates as specific in the dataset.
This is useful to store memory of specific events occurring on these dates ;
this is intended, e.g, to smart-sample the dataset at the point of training,
or to build a catalogue of such events along with the datasets.


For example:

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset("aifs-ea-an-oper-0001-mars-o48-2020-2021-6h-v1.zarr",
                     select="2t")

   ds.flagged()

   {'general': ['2020-08-12T00:00','2020-08-12T06:00', '2020-08-12T12:00']}


****************************
 Flagging dates in datasets
****************************

There are two ways to flag dates in anemoi-datasets:

   - either when building an anemoi-dataset ; the `flagged` key must be filled with the definition of the dates to be included
   - or as a metadata edit to an existing `.zarr` file, with the :ref:`flag` command.

For the moment, flagged dates are only implemented for regular dates, not trajectories or hindcasts.

Definition of flagged dates:

.. literalinclude:: ../yaml/flagged_dates.yaml
   :language: yaml

Or as a range of dates with start/end/frequency:

.. literalinclude:: ../yaml/flagged_dates_range.yaml
   :language: yaml


Note that ranges may overlap. Resulting dates from all ranges will be flagged only once.

You can subdivide flagged dates into multiple categories if needed.
By default, all flagged dates will end up in the `general` category.

If, e.g, you want to flag all heat waves and cold spells occurring for a given period, use something as:

.. literalinclude:: ../yaml/flagged_dates_temperature.yaml
   :language: yaml

Accessing each category can thus be done with:

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset("aifs-ea-an-oper-0001-mars-o48-2020-2021-6h-v1.zarr",
                     select="2t")

   ds.flagged('heatwaves')
