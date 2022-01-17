================
 API
================


Input data
------------------------

CMRR format
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: niphlem.input_data.load_cmrr_info

.. autofunction:: niphlem.input_data.load_cmrr_data

BIDS format
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: niphlem.input_data.load_bids_physio


Models
----------------

Retroicor
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: niphlem.models.RetroicorPhysio
   :members:
   :inherited-members:

Variation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: niphlem.models.RVPhysio
   :members:
   :inherited-members:


.. autoclass:: niphlem.models.HVPhysio
   :members:
   :inherited-members:

Downsample
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: niphlem.models.DownsamplePhysio
   :members:
   :inherited-members:


Report
----------------

.. autofunction:: niphlem.report.make_ecg_report
