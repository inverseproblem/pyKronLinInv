.. KronLinInv documentation master file, created by
   sphinx-quickstart on Mon Jul 15 21:57:41 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


     

``kronlininv``'s documentation
################################
	     
Overview
*************

This document describes the Python version of the code KronLinInv.

Kronecker-product-based linear inversion of geophysical (or other kinds of) data under Gaussian and separability assumptions. The code computes the posterior mean model and the posterior covariance matrix (or subsets of it) in an efficient manner (parallel algorithm) taking into account 3-D correlations both in the model parameters and in the observed data.
 
 If you use this code for research or else, please cite the related paper: 

Andrea Zunino, Klaus Mosegaard (2018), **An efficient method to solve large linearizable inverse problems under Gaussian and separability assumptions**, *Computers & Geosciences*. ISSN 0098-3004, <https://doi.org/10.1016/j.cageo.2018.09.005>.

See the above mentioned paper for a detailed description.



.. toctree::
   :caption: Contents:
   :maxdepth: 4
   
   userguide.rst
   apilist.rst
	     


Indices and tables
********************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



