:mod:`nff.data`
======================

.. automodule:: nff.data


Creating datasets
---------------------

.. autoclass:: nff.data.Dataset
   :members:

Joining and splitting the data from the dataset
-----------------------------------------------------

.. autofunction:: concatenate_dict
.. autofunction:: split_train_test
.. autofunction:: split_train_validation_test


Custom collate function for the DataLoader
------------------------------------------------
.. autofunction:: collate_dicts
