Installation
============

Install JAXSEDFit from the repository root:

.. code-block:: bash

   python -m pip install .

JAXSEDFit expects a DSPS SSP template file for host-galaxy modeling. A
continuum-only FSPS template is recommended because nebular emission lines are
modeled separately:

.. code-block:: bash

   curl -L -o tempdata.h5 https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_continuum_fsps_v3.2_lgmet_age.h5

Pass the template path through ``cfg.galaxy.dsps_ssp_fn``.

Milky Way dereddening requires ``dustmaps`` to be configured with local SFD
maps:

.. code-block:: bash

   python setup.py fetch --map-name=sfd
