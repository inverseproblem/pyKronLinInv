.. KronLinInv documentation master file, created by
   sphinx-quickstart on Mon Jul 15 21:57:41 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


``kronlininv``'s documentation
*********************************
	     
User guide
++++++++++++++++++

 This document describes the Python version of the code KronLinInv.

Kronecker-product-based linear inversion of geophysical (or other kinds of) data under Gaussian and separability assumptions. The code computes the posterior mean model and the posterior covariance matrix (or subsets of it) in an efficient manner (parallel algorithm) taking into account 3-D correlations both in the model parameters and in the observed data.
 
 If you use this code for research or else, please cite the related paper: 

Andrea Zunino, Klaus Mosegaard (2018), **An efficient method to solve large linearizable inverse problems under Gaussian and separability assumptions**, *Computers & Geosciences*. ISSN 0098-3004, <https://doi.org/10.1016/j.cageo.2018.09.005>.

See the above mentioned paper for a detailed description.


Theoretical background
============================

KronLinInv solves the *linear inverse problem* with Gaussian uncertainties
represented by the following objective function

.. math::

   S( \mathbf{m}) = \frac{1}{2} ( \mathbf{G} \mathbf{m} - \mathbf{d}_{\sf{obs}} )^{\sf{T}} \mathbf{C}^{-1}_{\rm{D}} ( \mathbf{G} \mathbf{m} - \mathbf{d}_{\sf{obs}} ) + \frac{1}{2} ( \mathbf{m} - \mathbf{m}_{\sf{prior}} )^{\sf{T}} \mathbf{C}^{-1}_{\rm{M}} ( \mathbf{m} - \mathbf{m}_{\sf{prior}} )


under the following separability conditions (for a 3-way decomposition):

.. math::
   \mathbf{C}_{\rm{M}} = \mathbf{C}_{\rm{M}}^{\rm{x}} \otimes 
   \mathbf{C}_{\rm{M}}^{\rm{y}} \otimes \mathbf{C}_{\rm{M}}^{\rm{z}} 
   , \quad
   \mathbf{C}_{\rm{D}} = \mathbf{C}_{\rm{D}}^{\rm{x}} \otimes 
   \mathbf{C}_{\rm{D}}^{\rm{y}} \otimes \mathbf{C}_{\rm{D}}^{\rm{z}} 
   \quad \textrm{ and } \quad
   \mathbf{G} = \mathbf{G}^{\rm{x}} \otimes \mathbf{G}^{\rm{y}} \otimes \mathbf{G}^{\rm{z}} \, .


From the above, the posterior covariance matrix is given by 

.. math::
   \mathbf{\widetilde{C}}_{\rm{M}} =  \left( \mathbf{G}^{\sf{T}} \,
   \mathbf{C}^{-1}_{\rm{D}} \, \mathbf{G} + \mathbf{C}^{-1}_{\rm{M}} \right)^{-1}

and the center of posterior gaussian is 

.. math::
   \mathbf{\widetilde{m}}  
   = \mathbf{m}_{\rm{prior}}+ \mathbf{\widetilde{C}}_{\rm{M}} \, \mathbf{G}^{\sf{T}} \, \mathbf{C}^{-1}_{\rm{D}} \left(\mathbf{d}_{\rm{obs}} - \mathbf{G} \mathbf{m}_{\rm{prior}} \right) \, .
   
KronLinInv solves the inverse problem in an efficient manner, with a very low memory imprint, suitable for large problems where many model parameters and observations are involved.

The paper describes how to obtain the solution to the above problem as shown hereafter. First the following matrices are computed

.. math::
   \mathbf{U}_1 \mathbf{\Lambda}_1  \mathbf{U}_1^{-1}  
   = \mathbf{C}_{\rm{M}}^{\rm{x}} (\mathbf{G}^{\rm{x}})^{\sf{T}}
   (\mathbf{C}_{\rm{D}}^{\rm{x}})^{-1} \mathbf{G}^{\rm{x}}

.. math::
   \mathbf{U}_2 \mathbf{\Lambda}_2  \mathbf{U}_2^{-1}
   =  \mathbf{C}_{\rm{M}}^{\rm{y}} (\mathbf{G}^{\rm{y}})^{\sf{T}}
   (\mathbf{C}_{\rm{D}}^{\rm{y}})^{-1} \mathbf{G}^{\rm{y}}


.. math::
   \mathbf{U}_3 \mathbf{\Lambda}_3  \mathbf{U}_3^{-1}
   = \mathbf{C}_{\rm{M}}^{\rm{z}} (\mathbf{G}^{\rm{z}})^{\sf{T}}
   (\mathbf{C}_{\rm{D}}^{\rm{z}})^{-1} \mathbf{G}^{\rm{z}}  \, .


The posterior covariance is then expressed as

.. math::
   \mathbf{\widetilde{C}}_{\rm{M}} = 
   \left(  
   \mathbf{U}_1 \otimes \mathbf{U}_2 \otimes \mathbf{U}_3 
   \right)
   \big( 
   \mathbf{I} + \mathbf{\Lambda}_1 \! \otimes \! \mathbf{\Lambda}_2 \! \otimes \! \mathbf{\Lambda}_3 
   \big)^{-1} 
   \big( 
   \mathbf{U}_1^{-1}  \mathbf{C}_{\rm{M}}^{\rm{x}} \otimes 
   \mathbf{U}_2^{-1} \mathbf{C}_{\rm{M}}^{\rm{y}} \otimes 
   \mathbf{U}_3^{-1} \mathbf{C}_{\rm{M}}^{\rm{z}} 
   \big) \, .

and the posterior mean model as

.. math::
   \mathbf{\widetilde{m}} =  
   \mathbf{m}_{\rm{prior}} +  
   \Big[ \!
   \left(  
   \mathbf{U}_1 \otimes \mathbf{U}_2 \otimes \mathbf{U}_3 
   \right)
   \big( 
   \mathbf{I} + \mathbf{\Lambda}_1\!  \otimes \! \mathbf{\Lambda}_2 \!  \otimes\!  \mathbf{\Lambda}_3 
   \big)^{-1} \\ 
   \times \Big( 
   \left( \mathbf{U}_1^{-1}  \mathbf{C}_{\rm{M}}^{\rm{x}} (\mathbf{G}^{\rm{x}})^{\sf{T}} (\mathbf{C}_{\rm{D}}^{\rm{x}})^{-1} \right) \!    \otimes 
   \left( \mathbf{U}_2^{-1} \mathbf{C}_{\rm{M}}^{\rm{y}}  (\mathbf{G}^{\rm{y}})^{\sf{T}} (\mathbf{C}_{\rm{D}}^{\rm{y}})^{-1}  \right)   \!   
   \\ 
   \otimes  \left( \mathbf{U}_3^{-1} \mathbf{C}_{\rm{M}}^{\rm{z}} (\mathbf{G}^{\rm{z}})^{\sf{T}} (\mathbf{C}_{\rm{D}}^{\rm{z}})^{-1} \right)
   \Big)
   \Big] \\
   \times \Big( \mathbf{d}_{\rm{obs}} - \big( \mathbf{G}^{\rm{x}} \otimes \mathbf{G}^{\rm{y}} \otimes \mathbf{G}^{\rm{z}} \big) \, \mathbf{m}_{\rm{prior}} \Big) \, .

These last two formulae are those used by the KronLinInv algorithm.

Several function are exported by the module KronLinInv:

- :func:`calcfactors()`: Computes the factors necessary to solve the inverse problem

- :func:`posteriormean()`: Computes the posterior mean model using the previously computed "factors" with :func:`calcfactors()`.

- :func:`blockpostcov()`: Computes a block (or all) of the posterior covariance using the previously computed "factors" with :func:`calcfactors()`.

- :func:`bandpostcov()`: NOT YET IMPLEMENTED! Computes a band of the posterior covariance the previously computed "factors" with :func:`calcfactors()`.



Usage examples
===================

The input needed is represented by the set of three covariance matrices of the model parameters, the three covariances of the observed data, the three forward model operators, the observed data (a vector) and the prior model (a vector).
_The **first** thing to compute is always the set of "factors" using the function :func:`calcfactors()`. 
Finally, the posterior mean (see :func:`posteriormean()`) and/or covariance (or part of it) (see :func:`blockpostcov()`) can be computed.

2D example
+++++++++++++

An example of how to use the code for 2D problems is shown in the following. Notice that the code is written for a 3D problem, however, by setting some of the matrices as identity matrices with size of 1×1, a 2D problem can be solved without much overhead.


Creating a test problem
------------------------

First, we create some input data to simulate a real problem.

.. code::
   
    ## 2D problem, so set nx = 1
    nx = 1
    ny = 20
    nz = 30
    nxobs = 1
    nyobs = 18
    nzobs = 24

We then construct some covariance matrices and a forward operator. The "first" covariance matrices for model parameters
(:math:`\mathbf{C}_{\rm{M}}^{\rm{x}} \, , \mathbf{C}_{\rm{M}}^{\rm{y}} \, , \mathbf{C}_{\rm{M}}^{\rm{z}}`) and observed data (:math:`\mathbf{C}_{\rm{D}}^{\rm{x}} \, , \mathbf{C}_{\rm{D}}^{\rm{y}} \, , \mathbf{C}_{\rm{D}}^{\rm{z}}`) are simply an identity matrix of shape 1``\times``1, since it is a 2D problem.
The forward relation (forward model) is created from three operators (:math:`\mathbf{G}^{\rm{x}} \, , \mathbf{G}^{\rm{y}} \, , \mathbf{G}^{\rm{z}}`). Remark: the function :func:`mkCovSTAT()` used in the following example is *not* part of KronLinInv.

.. code::

 def mkCovSTAT(sigma,nx,ny,nz,corrlength,kind) :
    #
    #   Stationaly covariance model 
    #

    def cgaussian(dist,corrlength):
        if dist.max()==0.0:
            return 1.0
        else:
            assert(corrlength>0.0)
            return np.exp(-(dist/corrlength)**2)
        
    def cexponential(dist,corrlength):
        if dist.max()==0.0:
            return 1.0
        else:
            assert(corrlength>0.0)
            return np.exp(-(dist/corrlength))
    
    npts = nx*ny*nz
    x = np.asarray([float(i) for i in range(nx)])
    y = np.asarray([float(i) for i in range(ny)])
    z = np.asarray([float(i) for i in range(nz)])
    covmat_x = np.zeros((nx,nx))
    covmat_y = np.zeros((ny,ny))
    covmat_z = np.zeros((nz,nz))

    if kind=="gaussian" :
        calccovfun = cgaussian
    elif kind=="exponential" :
        calccovfun = cexponential
    else :
        print("Error, no or wrong cov 'kind' specified")
        raise ValueError

    for i in range(nx):
        covmat_x[i,:] = sigma[0]**2 * calccovfun(np.sqrt((x-x[i])**2),corrlength[0])

    for i in range(ny):
        covmat_y[i,:] = sigma[1]**2 * calccovfun(np.sqrt(((y-y[i]))**2),corrlength[1])

    for i in range(nz):
        covmat_z[i,:] = sigma[2]**2 * calccovfun(np.sqrt(((z-z[i]))**2),corrlength[2])
    
    return covmat_x,covmat_y,covmat_z

.. code::

   sigmaobs  = np.array([1.0, 0.1, 0.1])
   corlenobs = np.array([0.0, 1.4, 1.4])
   sigmam    = np.array([1.0, 0.8, 0.8])
   corlenm   = np.array([0.0, 2.5, 2.5])

   ## Covariance matrices
   # covariance on observed data
   Cd1,Cd2,Cd3 = mkCovSTAT(sigmaobs,nxobs,nyobs,nzobs,corlenobs,"gaussian")
   # covariance on model parameters
   Cm1,Cm2,Cm3 = mkCovSTAT(sigmam,nx,ny,nz,corlenm,"gaussian") 

   ## Forward model operator
   G1 = np.random.rand(nxobs,nx)
   G2 = np.random.rand(nyobs,ny)
   G3 = np.random.rand(nzobs,nz)


Finally, a "true/reference" model, in order to compute some synthetic "observed" data and a prior model.

.. code::
   
   ## Create a reference model
   refmod = np.random.rand(nx*ny*nz)

   ## Create a reference model
   mprior = refmod.copy() #0.5 .* ones(nx*ny*nz)

   ## Create some "observed" data
   ##   (without noise because it's just a test of the algo)
   dobs = np.kron(G1,np.kron(G2,G3)) @ refmod 

Now we have create a synthetic example to play with, which we can solve as shown in the following.


Solving the 2D problem
------------------------

In order to solve the inverse problem using KronLinInv, we first need to compute the "factors" using the function :func:`calcfactors()`, which takes as inputs two `struct`s containing the covariance matrices and the forward operators.

.. code::

   ## Calculate the required factors
   klifac = kli.calcfactors(G1,G2,G3,Cm1,Cm2,Cm3,Cd1,Cd2,Cd3)

Now the inverse problem can be solved. We first compute the posterior mean and then a subset of the posterior covariance.

.. code::

   ## Calculate the posterior mean model
   postm = kli.posteriormean(klifac,G1,G2,G3,mprior,dobs)

   ## Calculate the posterior covariance
   npts = nx*ny*nz
   astart, aend = 0,npts//3
   bstart, bend = 0,npts//3
   postC = kli.blockpostcov(klifac,astart,aend,bstart,bend)




3D example
++++++++++++++++

.. Example


======================================
API ``kronlininv``, list of functions
======================================

.. automodule:: kronlininv
   :members:
   :imported-members: 


.. toctree::
   :maxdepth: 3
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
