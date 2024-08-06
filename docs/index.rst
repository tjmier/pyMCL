.. pyMCL documentation master file, created by
   sphinx-quickstart on Mon Aug  5 10:01:33 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The pyMCL Package
=================================

Introduction
************
``pyMCL`` is an open-source Python package for the implimentaion of the markov cluster algorithm. 

The package is designed to be easy to use and flexible, allowing users to easily implement the MCL algorithm on their own data as either ``numpy.array`` or ``pandas.DataFrame``.

The orginal MCL algorithm was developed by Stijn van Dongen in 2000, and has since been used in a variety of applications, including the analysis of biological networks, social networks, and more.

Motivation
***********

Although there are many other implementaions of the MCL algortim in other languages, currently, there are no functional Python packages that provide a way to implement the MCL algorithm. The aim of ``pyMCL`` is to provide as simple implementation of the MCL algorithm that can be easily intergrated into python data analysis pipelines. Compatability with ``NumPy`` and ``Pandas`` data structures is a key feature of this package due to the popularity of these libraries in the data science community.

View the `source code <https://github.com/tjmier/pyMCL>`_ of pyMCL.

Limitations
************

The primary limitation of ``pyMCL`` is that the MCL algorithm is a computationally expensive algorithm due to the fact that it applies ~10-100 instances of matrix multiplication. The matrix is of size NxN where N is the number of nodes in the graph. This results in N^3 calculations per matrix multipication. As such, user should be aware that the computation time will increase drastically as number of nodes and non-zero edges increases. **This is an inherant limiation of the MCL algorithm.** 

Users are encouraged to try the original C++ implementation of the MCL algorithm if they require faster performance.

Utilization of GPU parrell processing with ``CuPy`` or ``Numba`` would dramatically improve the performance of ``pyMCL`` (or any other implementation of MCL), therefore if there is interest in the project, this feature will be added in the future.

.. toctree::
   :maxdepth: 2
   :caption: Tutorial:

   Installation
   Quickstart

.. toctree::
   :maxdepth: 2
   :caption: Background:

   intro

.. toctree::
   :maxdepth: 2
   :caption: Refrence:

   pyMCL

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
