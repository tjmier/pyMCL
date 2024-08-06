Installation
============

Requirements
*******************

``pyMCL`` requires only two Python packages:

- `NumPy <https://numpy.org/>`_, for all matrix operations
- `Pandas <https://pandas.pydata.org/>`_, for returning named incides 

``pyMCL`` is supported on Linux, macOS and Windows on Python 3.9 to 3.12.

Using Conda
*******************

.. ipython::

    In [136]: x = 2

    In [137]: x**3
    Out[137]: 8

.. code-block:: bash

    $ conda install -c conda-forge pyMCL

Alternatively, mamaba can be used for a faster installation:

.. code-block:: bash

    $ conda install -c conda-forge mamba
    $ mamba install -c conda-forge pyMCL

Using pip
*******************

``pyMCL`` is avaible on PyPI and can be installed using pip:

.. code-block:: bash

    $ pip install pyMCL

You can also install the latest version of ``pyMCL`` from the source code on GitHub:

.. code-block:: bash

    $ pip install git+https://github.com/tjmier/pyMCL.git

.. note::

    Installing the lastest version from the source code allows you to access the latest features and bug fixes, but may not be as stable as the version on PyPI or conda.
    If you do come across any issues or bugs, please report them on the `GitHub issues page <https://github.com/tjmier/pyMCL/issues>`_.