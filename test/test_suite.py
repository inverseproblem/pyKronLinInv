


import sys
sys.path.append("../")

import numpy as np
import kronlininv as kli


###############################################

def test2D( ):

    ## 2D problem, so set nx = 1
    nx = 1
    ny = 20
    nz = 30
    nxobs = 1
    nyobs = 18
    nzobs = 24
    sigmaobs  = np.array([1.0, 0.1, 0.1])
    corlenobs = np.array([0.0, 1.4, 1.4])
    sigmam    = np.array([1.0, 0.8, 0.8])
    corlenm   = np.array([0.0, 2.5, 2.5])
    
    # run test 
    result = calc_mean_cov(nx,ny,nz,nxobs,nyobs,nzobs,
                           sigmaobs,corlenobs,sigmam,corlenm)

    return result


#########################################
# 
# pytest considers all functions starting with test 
#  to be tests to be run
#
#
#########################################

def test3D( ):

    ## 2D problem, so set nx = 1
    nx = 8
    ny = 9
    nz = 12
    nxobs = 6
    nyobs = 7
    nzobs = 11
    sigmaobs  = np.array([0.1, 0.1, 0.1])
    corlenobs = np.array([1.4, 1.4, 1.4])
    sigmam    = np.array([0.8, 0.8, 0.8])
    corlenm   = np.array([2.5, 2.5, 2.5])

    
    # run test 
    result = calc_mean_cov(nx,ny,nz,nxobs,nyobs,nzobs,
                           sigmaobs,corlenobs,sigmam,corlenm)

    return result

########################################

def calc_mean_cov(nx,ny,nz,nxobs,nyobs,nzobs,sigmaobs,corlenobs,
                  sigmam,corlenm) :

    ##############################
    ## Setup the problem
    ##############################

    ## Covariance matrices
    # covariance on observed data
    Cd1,Cd2,Cd3 = mkCovSTAT(sigmaobs,nxobs,nyobs,nzobs,corlenobs,"gaussian")
    # covariance on model parameters
    Cm1,Cm2,Cm3 = mkCovSTAT(sigmam,nx,ny,nz,corlenm,"gaussian") 

    ## Forward model operator
    G1 = np.random.rand(nxobs,nx)
    G2 = np.random.rand(nyobs,ny)
    G3 = np.random.rand(nzobs,nz)

    ##############################
    ## Setup the synthetic test
    ##############################

    ## Create a reference model
    refmod = np.random.rand(nx*ny*nz)

    ## Create a reference model
    mprior = refmod.copy() #0.5 .* ones(nx*ny*nz)

    ## Create some "observed" data
    ##   (without noise because it's just a test of the algo)
    dobs = np.kron(G1,np.kron(G2,G3)) @ refmod 


    ##############################
    ## Solve the inverse problem
    ##############################

    ## Calculate the required factors
    klifac = kli.calcfactors(G1,G2,G3,Cm1,Cm2,Cm3,Cd1,Cd2,Cd3)

    ## Calculate the posterior mean model
    postm = kli.posteriormean(klifac,G1,G2,G3,mprior,dobs)

    ## Calculate the posterior covariance
    npts = nx*ny*nz
    astart, aend = 0,npts//3
    bstart, bend = 0,npts//3
    postC = kli.blockpostcov(klifac,astart,aend,bstart,bend)

    ##############################
    ## Check results
    ##############################

    if all(np.abs(postm-refmod)<=1e-3):
        return True
    return False

###########################################################

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

##########################################################

if __name__=="__main__" :

    res=test2D()
    print("\nTest 2D passed: {}\n".format(res))
    res=test3D()
    print("\nTest 3D passed: {}\n".format(res))
