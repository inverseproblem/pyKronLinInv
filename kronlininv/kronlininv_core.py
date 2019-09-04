
#------------------------------------------------------------------------
#
#    Copyright 2019, Andrea Zunino 
#
#    This file is part of Kronlininv.
#
#    Kronlininv is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Kronlininv is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Kronlininv.  If not, see <http://www.gnu.org/licenses/>.
#
#------------------------------------------------------------------------

from numba import jit
import numpy as np
import scipy.linalg as la
import sys
from collections import namedtuple

#assert sys.version_info >= (3, 5)

##########################################################
##########################################################

def calcfactors( G1,G2,G3,Cm1,Cm2,Cm3,Cd1,Cd2,Cd3 ) :     
    """
     Compute the factors required to solve the inverse problem.
     These factors are the ones to be stored to compute the posterior covariance or mean.

    Args:
        G1,G2,G3 (ndarrays): the three forward model matrices 
        Cm1,Cm2,Cm3 (ndarrays): the three covariance matrices related to the model parameters
        Cd1,Cd2,Cd3 (ndarrays): the three covariance matrices related to the observed data

    Returns:
        named tuple: a set of arrays required to subsequently calculate the posterior mean or covariance
    
    """
     
    print('Preliminary checks')
    lsm = [Cm1,Cm2,Cm3,Cd1,Cd2,Cd3]
    lsn = ["Cm1","Cm2","Cm3","Cd1","Cd2","Cd3"]
    for i,mat in enumerate(lsm) :
        checkmat( mat,lsn[i] )
          
    print('Calculating preliminary stuff')
    iCdG1 = la.solve(Cd1,G1, sym_pos=True)
    iCdG2 = la.solve(Cd2,G2, sym_pos=True)
    iCdG3 = la.solve(Cd3,G3, sym_pos=True)
    GtiCd1  = np.dot( G1.T, iCdG1 )
    GtiCd2  = np.dot( G2.T, iCdG2 ) 
    GtiCd3  = np.dot( G3.T, iCdG3 )

    ######################################
    ###  Eigendecomposition in Python    #
    ######################################
    print('Calculating eigendecomposition')
    lambda1,U1 = la.eigh(GtiCd1,b=Cm1, lower=False, eigvals_only=False,
                         overwrite_a=False, overwrite_b=False, turbo=True,
                         eigvals=None, type=3, check_finite=True)
    lambda2,U2 = la.eigh(GtiCd2,b=Cm2, lower=False, eigvals_only=False,
                         overwrite_a=False, overwrite_b=False, turbo=True,
                         eigvals=None, type=3, check_finite=True)
    lambda3,U3 = la.eigh(GtiCd3,b=Cm3, lower=False, eigvals_only=False,
                         overwrite_a=False, overwrite_b=False, turbo=True,
                         eigvals=None, type=3, check_finite=True)

    ###################################
    ### calculating the 3 factors

    print('Calculating fa')
    n1=lambda1.size
    n2=lambda2.size
    n3=lambda3.size

    print('Calculating fb')
    argfb = 1.00 + krondiag(lambda1,krondiag(lambda2,lambda3))
    vecdfac = 1.00/argfb
    
    print('Calculating fc')
    iUCm1 = la.solve(U1,Cm1) 
    iUCm2 = la.solve(U2,Cm2) 
    iUCm3 = la.solve(U3,Cm3) 

    print('Calculating fd')
    fd11 = la.solve( Cd1, np.identity(Cd1.shape[0])) 
    fd22 = la.solve( Cd2, np.identity(Cd2.shape[0])) 
    fd33 = la.solve( Cd3, np.identity(Cd3.shape[0])) 
    iUCmGtiCd1 = iUCm1 @ (G1.T @ fd11)
    iUCmGtiCd2 = iUCm2 @ (G2.T @ fd22)
    iUCmGtiCd3 = iUCm3 @ (G3.T @ fd33)
          
    FactorsKLI = namedtuple('FactorsKLI',['U1','U2','U3','invlambda','iUCm1','iUCm2','iUCm3','iUCmGtiCd1','iUCmGtiCd2','iUCmGtiCd3'])

    factkli = FactorsKLI(U1,U2,U3, vecdfac, iUCm1,iUCm2,iUCm3, iUCmGtiCd1,iUCmGtiCd2,iUCmGtiCd3)

    return factkli

##########################################################

@jit(parallel=True) #(nopython=True,parallel=True)
def posteriormean(factkli, G1,G2,G3, mprior, dobs ) :
    """
    Compute the posterior mean model.

    Args: 
        factkli (named tuple): a set of arrays computed with the function calcfactors()
        G1,G2,G3 (ndarrays): the three forward model matrices 
        mprior (ndarray, 1D): the prior model
        dobs (ndarray, 1D): the observed data

    Returns:
        ndarray, 1D: the posterior mean model

    """      

    U1,U2,U3 = factkli.U1,factkli.U2,factkli.U3
    diaginvlambda = factkli.invlambda
    Z1,Z2,Z3 = factkli.iUCmGtiCd1,factkli.iUCmGtiCd2,factkli.iUCmGtiCd3

    # sizes
    Na = mprior.size
    Nb = dobs.size
    Ni = Z1.shape[0]
    Nl = Z1.shape[1]
    Nj = Z2.shape[0]
    Nm = Z2.shape[1]
    Nk = Z3.shape[0]
    Nn = Z3.shape[1]

    ## allocate stuff
    ddiff = np.zeros(Nb)
    Zh = np.zeros(Na)
    elUDZh = np.zeros(Na)

    av = np.arange(0,Na) 
    bv = np.arange(0,Nb) 

    iv = av//(Nk*Nj) 
    jv = (av-iv*Nk*Nj)//Nk 
    kv = av-jv*Nk-iv*Nk*Nj 
    
    lv = bv//(Nn*Nm) 
    mv = (bv-lv*Nn*Nm)//Nn 
    nv = bv-mv*Nn-lv*Nn*Nm
    
    ##---------------------
    if Nb>20 :
        everynit = Nb//20
    else :
        everynit = 1

    postm = np.zeros(Na)
    
    for b in range(Nb) :
        ddiff[b] = dobs[b] - np.sum( mprior * G1[lv[b],iv] * G2[mv[b],jv] * G3[nv[b],kv] )

    for a in range(Na) :
        if a%everynit==0 :
            line = 'postmean(): 1st loop {} of {}   \r'.format(a,Na)
            sys.stdout.write(line)
            sys.stdout.flush()

        Zh[a] = np.sum( Z1[iv[a],lv] * Z2[jv[a],mv] * Z3[kv[a],nv] * ddiff )


    for a in range(Na) :
        if a%everynit==0 :
            line = 'postmean(): 2nd loop {} of {}   \r'.format(a,Na)
            sys.stdout.write(line)
            sys.stdout.flush()
       
        elUDZh[a] = np.sum( U1[iv[a],iv] * U2[jv[a],jv] * U3[kv[a],kv]  * diaginvlambda * Zh )
      
        # element of the posterior mean
        postm[a] = mprior[a] + elUDZh[a]       

    return postm

##########################################################

@jit(parallel=True) #(nopython=True,parallel=True)
def blockpostcov(factkli, astart,aend,bstart,bend ) :

    """
    Compute a block of the posterior covariance defined by the min/max indices astart/aend (rows), bstart/bend (columns).

    Args:
        factkli: (named tuple): a set of arrays computed with the function calcfactors()
        astart,aend (integers): the first and last row of the posterior covariance to be computed
        bstart,bend (integers): the first and last column of the posterior covariance to be computed

    Returns:
        ndarray: a block of the posterior covariance 

    """

    assert(aend>=astart)
    assert(bend>=bstart)

    U1,U2,U3 = factkli.U1,factkli.U2,factkli.U3
    diaginvlambda = factkli.invlambda
    iUCm1,iUCm2,iUCm3 = factkli.iUCm1,factkli.iUCm2,factkli.iUCm3
    
    s1 = aend+1-astart
    s2 = bend+1-bstart
    
    postC = np.zeros((s1,s2))

    # sizes
    Ni = U1.shape[0]
    Nl = U1.shape[1]
    Nj = U2.shape[0]
    Nm = U2.shape[1]
    Nk = U3.shape[0]
    Nn = U3.shape[1]
    Na = Ni*Nj*Nk
    Nb = Nl*Nm*Nn

    ## -----
    av = np.arange(0,Na) 
    bv = np.arange(0,Nb) 

    iv = av//(Nk*Nj) 
    jv = (av-iv*Nk*Nj)//Nk 
    kv = av-jv*Nk-iv*Nk*Nj 
    
    lv = bv//(Nn*Nm) 
    mv = (bv-lv*Nn*Nm)//Nn 
    nv = bv-mv*Nn-lv*Nn*Nm

    ## -----
    if s1>20 :
        everynit = s1//20
    else :
        everynit = 1

    for a in range(astart,aend+1)  :
        
        if a%everynit==0 :
            line = 'blockpostcov(): loop {} of {}   \r'.format(a,aend)
            sys.stdout.write(line)
            sys.stdout.flush()

        # row first two factors
        # a row x diag matrix 
        row2 =  U1[iv[a],iv] * U2[jv[a],jv] * U3[kv[a],kv] * diaginvlambda
        
        for b in range(bstart,bend+1) :
            
            ## calculate one row of first TWO factors
            col1 = iUCm1[iv,iv[b]] * iUCm2[jv,jv[b]] * iUCm3[kv,kv[b]]
            
            ## calculate one element of the posterior covariance
            postC[a,b] = np.sum(row2*col1)

    return postC

##########################################################
  
@jit(nopython=True)
def krondiag(a,b) :
    """
      Kronecker product of two diagonal matrices.
      Returns only the diagonal as a vector.
    """
    assert a.ndim==1
    assert b.ndim==1
    ni=a.size
    nj=b.size
    c = np.zeros(ni*nj,dtype=a.dtype) ###,dtype=np.complex128)
    k=0
    for i in range(ni) :
        c[k:k+nj] = a[i]*b
        k += nj
    return c

##########################################################
        
def checkmat( mat,matname ):
    """
     Check certain properties of the input matrix.
    """
    print('Checking {}'.format(matname))
    issym = np.allclose(mat,mat.T)
    if not issym :
        print('Matrix {} is not symmetric'.format(matname))
        raise ValueError 
    try:
        np.linalg.cholesky(mat)
    except np.linalg.linalg.LinAlgError:
        print('Matrix {} is not positive definite'.format(matname))
        raise ValueError 
    return True
    
##########################################################


