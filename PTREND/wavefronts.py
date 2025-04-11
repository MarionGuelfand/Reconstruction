import numpy as np
import sys
import pandas as pd
from numba import njit, float64, prange
from scipy.spatial.transform import Rotation as R
from scipy.optimize import fsolve
from solver import newton
from rotation import rotation
from scipy.optimize import fsolve, brentq
import argparse


# Used for interpolation
n_omega_cr = 20

# Physical constants
c_light = 2.997924580e8
R_earth = 6371007.0
ns = 325
kr = -0.1218
    
if len(sys.argv) > 3:
    groundAltitude = float(sys.argv[3])
else:
    groundAltitude = 1086  


B_dec = 0.
B_inc = np.pi/2. + 1.0609856522873529
# Magnetic field direction (unit) vector
Bvec = np.array([np.sin(B_inc)*np.cos(B_dec),np.sin(B_inc)*np.sin(B_dec),np.cos(B_inc)])

kwd = {"fastmath": {"reassoc", "contract", "arcp"}}

# Simple numba example
@njit(**kwd)
def dotme(x,y,z):
    res =  np.dot(x,x)
    res += np.dot(y,y)
    res += np.dot(z,z)
    return(res)

@njit(**kwd)
def RefractionIndexAtPosition(X):

    R2 = X[0]*X[0] + X[1]*X[1]
    h = (np.sqrt( (X[2]+R_earth)**2 + R2 ) - R_earth)/1e3 # Altitude in km
    rh = ns*np.exp(kr*h)
    n = 1.+1e-6*rh
    #n = 1.+(1e-6*rh)/2
    return (n)

@njit(**kwd)
def ZHSEffectiveRefractionIndex(X0,Xa):

    R02 = X0[0]**2 + X0[1]**2
    
    # Altitude of emission in km
    h0 = (np.sqrt( (X0[2]+R_earth)**2 + R02 ) - R_earth)/1e3
    # print('Altitude of emission in km = ',h0)
    # print(h0)
    
    # Refractivity at emission 
    rh0 = ns*np.exp(kr*h0)

    modr = np.sqrt(R02)
    # print(modr)

    if (modr > 1e3):

        # Vector between antenna and emission point
        U = Xa-X0
        # Divide into pieces shorter than 10km
        #nint = np.int(modr/2e4)+1
        nint = int(modr/2e4)+1
        K = U/nint

        # Current point coordinates and altitude
        Curr  = X0
        currh = h0
        s = 0.

        for i in np.arange(nint):
            Next = Curr + K # Next point
            nextR2 = Next[0]*Next[0] + Next[1]*Next[1]
            nexth  = (np.sqrt( (Next[2]+R_earth)**2 + nextR2 ) - R_earth)/1e3
            if (np.abs(nexth-currh) > 1e-10):
                s += (np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
            else:
                s += np.exp(kr*currh)

            Curr = Next
            currh = nexth
            # print (currh)

        avn = ns*s/nint
        # print(avn)
        n_eff = 1. + 1e-6*avn # Effective (average) index

    else:

        # without numerical integration
        hd = Xa[2]/1e3 # Antenna altitude
        #if (np.abs(hd-h0) > 1e-10):
        avn = (ns/(kr*(hd-h0)))*(np.exp(kr*hd)-np.exp(kr*h0))
        #else:
        #    avn = ns*np.exp(kr*h0)

        n_eff = 1. + 1e-6*avn # Effective (average) index

    return (n_eff)

@njit(**kwd)
def compute_observer_position(omega,Xmax,U,K, xmaxDist,alpha):
    '''
    Given angle between shower direction (K) and line joining Xmax and observer's position,
    horizontal direction to observer's position, Xmax position and groundAltitude, compute
    coordinates of observer
    '''

    # Compute rotation axis. Make sure it is normalized
    Rot_axis = np.cross(U,K)
    Rot_axis /= np.linalg.norm(Rot_axis)
    # Compute rotation matrix from Rodrigues formula
    Rotmat = rotation(-omega,Rot_axis)
    # Define rotation using scipy's method
    # Rotation = R.from_rotvec(-omega * Rot_axis)
    # print('#####')
    # print(Rotation.as_matrix())
    # print('#####')
    # Dir_obs  = Rotation.apply(K)
    Dir_obs = np.dot(Rotmat,K)
    # Compute observer's position
    # this assumed coincidence was computed at antenna altitude)
    # t = (Xant[2] - Xmax[2])/Dir_obs[2]
    # This assumes coincidence is computed at fixed alpha, i.e. along U, starting from Xcore
    t = np.sin(alpha)/np.sin(alpha+omega) * xmaxDist
    X = Xmax + t*Dir_obs
    return (X)

@njit(**kwd)
def minor_equation(omega, n2, n1, alpha, delta, xmaxDist):

    '''
    Compute time delay (in m)
    '''
    Lx = xmaxDist
    sa = np.sin(alpha)
    #saw = np.sin(alpha-omega) # Keeping minus sign to compare to Valentin's results. Should be plus sign.
    saw = np.sin(alpha+omega)
    com = np.cos(omega)
    l0 = Lx*sa/saw
    l1 = np.sqrt(l0**2+delta**2+2*delta*l0*com)
    l2 = np.sqrt(l0**2+delta**2-2*delta*l0*com)
    # Eq. 3.38 p125.
    res = (n2*l2+2*delta)**2-(n1*l1)**2
    #res = Lx*Lx * sa*sa *(n0*n0-n1*n1) + 2*Lx*sa*saw*delta*(n0-n1*n1*com) + delta*delta*(1.-n1*n1*com*com)*saw*saw
    return(res)

@njit(**kwd)
def master_equation(omega, n2, n1, alpha, delta, xmaxDist):

    '''
    Compute [c*delta(t)]^2    
    '''
    Lx = xmaxDist
    sa = np.sin(alpha)
    #saw = np.sin(alpha-omega) # Keeping minus sign to compare to Valentin's results. Should be plus sign.
    saw = np.sin(alpha+omega)
    com = np.cos(omega)
    l0 = Lx*sa/saw
    l1 = np.sqrt(l0**2+delta**2+2*delta*l0*com)
    l2 = np.sqrt(l0**2+delta**2-2*delta*l0*com)
    # Eq. 3.38 p125.
    res = (n2*l2-n1*l1+2*delta)**2
    #res = Lx*Lx * sa*sa *(n0*n0-n1*n1) + 2*Lx*sa*saw*delta*(n0-n1*n1*com) + delta*delta*(1.-n1*n1*com*com)*saw*saw
    return(res)

# Loss functions (chi2), according to different models:
# PWF: Plane wave function
# SWF: Spherical wave function
# ADF: Amplitude Distribution Function (see Valentin Decoene's thesis)

@njit(**kwd)
def PWF_loss(params, Xants, tants, verbose=False, cr=1.0):
    '''
    Defines Chi2 by summing model residuals
    over antenna pairs (i, j):
    loss = \sum_{i>j} ((Xants[i, :]-Xants[j, :]).K - cr(tants[i]-tants[j]))**2
    where:
    params=(theta, phi): spherical coordinates of unit shower direction vector K
    Xants are the antenna positions (shape=(nants, 3))
    tants are the antenna arrival times of the wavefront (trigger time, shape=(nants, ))
    cr is radiation speed, by default 1 since time is expressed in m.
    '''

    theta, phi = params
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp, st*sp, ct])
    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None
    # Use numpy outer methods to build matrix X_ij = x_i -x_j
    xk = np.dot(Xants, K)
    DXK = np.subtract.outer(xk, xk)
    DT  = np.subtract.outer(tants, tants)
    chi2 = ( (DXK - cr*DT)**2 ).sum() / 2. # Sum over upper triangle, diagonal is zero because of antisymmetry of DXK, DT
    if verbose:
        print("params = ", np.rad2deg(params))
        print("Chi2 = ", chi2)
    return(chi2)

@njit(**kwd)
def PWF_alternate_loss(params, Xants, tants, verbose=False, cr=1.0):
    '''
    Defines Chi2 by summing model residuals over individual antennas, 
    after maximizing likelihood over reference time.
    '''
    nants = tants.shape[0]
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None
    # Make sure tants and Xants are compatible
    residuals = PWF_residuals(params, Xants, tants, verbose=verbose, cr=cr)
    chi2 = (residuals**2).sum()
    return(chi2)

def PWF_minimize_alternate_loss(Xants, tants, verbose=False, cr=1.0):
    '''
    Solves the minimization problem by using a special solution to the linear regression
    on K(\theta, \phi), with the ||K||=1 constraint. Note that this is a non-convex problem.
    This is formulated as 
    argmin_k k^T.A.k - 2 b^T.k, s.t. ||k||=1
    '''
    nants = tants.shape[0]

    # Make sure tants and Xants are compatible

    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None
    # Compute A matrix (3x3) and b (3-)vector, see above
    PXT = Xants - Xants.mean(axis=0)  # P is the centering projector, XT=Xants
    A = np.dot(Xants.T, PXT)
    b = np.dot(Xants.T, tants-tants.mean(axis=0)) 
    # Diagonalize A, compute projections of b onto eigenvectors
    d, W = np.linalg.eigh(A)
    beta = np.dot(b, W)
    nbeta = np.linalg.norm(beta)

    if (np.abs(beta[0]/nbeta) < 1e-14):
        if (verbose):
            print("Degenerate case")
        # Degenerate case. This will be triggered e.g. when all antennas lie in a single plane.
        mu = -d[0]
        c = np.zeros(3)
        c[1] = beta[1]/(d[1]+mu)
        c[2] = beta[2]/(d[2]+mu)
        si = np.sign(np.dot(W[:, 0], np.array([0, 0, 1.])))
        c[0] = -si*np.sqrt(1-c[1]**2-c[2]**2)  # Determined up to a sign: choose descending solution
        k_opt = np.dot(W, c)
        # k_opt[2] = -np.abs(k_opt[2]) # Descending solution

    else:
        # Assume non-degenerate case, i.e. projections on smallest eigenvalue are non zero
        # Compute \mu such that \sum_i \beta_i^2/(\lambda_i+\mu)^2 = 1, using root finding on mu
        def nc(mu):
            # Computes difference of norm of k solution to 1. Coordinates of k are \beta_i/(d_i+\mu) in W basis
            c = beta/(d+mu)
            return ((c**2).sum()-1.)
        mu_min = -d[0]+beta[0]
        mu_max = -d[0]+np.linalg.norm(beta)
        mu_opt = brentq(nc, mu_min, mu_max, maxiter=1000)
        # Compute coordinates of k in W basis, return k
        c = beta/(d+mu_opt)
        k_opt = np.dot(W, c)

    # Now get angles from k_opt coordinates
    if k_opt[2] > 1e-2:
        k_opt = k_opt-2*(k_opt@W[:, 0])*W[:, 0]

    theta_opt = np.arccos(-k_opt[2])
    phi_opt = np.arctan2(-k_opt[1], -k_opt[0])

    if phi_opt < 0:
        phi_opt += 2*np.pi

    return (np.array([theta_opt, phi_opt]))


@njit(**kwd)
def PWF_residuals(params, Xants, tants, verbose=False, cr=1.0):

    '''
    Computes timing residuals for each antenna using plane wave model
    Note that this is defined at up to an additive constant, that when minimizing
    the loss over it, amounts to centering the residuals.
    '''
    nants = tants.shape[0]
    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None

    times = PWF_model(params, Xants, cr=cr)
    res = cr * (tants - times)
    res -= res.mean()  # Mean is projected out when maximizing likelihood over reference time t0
    return (res)

@njit(**kwd)
def PWF_simulation(params, Xants, sigma_t = 5e-9, iseed=None, cr=1.0):
    '''
    Generates plane wavefront timings, zero at shower core, with jitter noise added
    '''

    times = PWF_model(params,Xants,cr=cr)
    # Add noise
    if (iseed is not None):
        np.random.seed(iseed)
    n = np.random.standard_normal(times.size) * sigma_t * c_light
    return (times + n)

@njit(**kwd)
def PWF_model(params, Xants, cr=1.0):
    '''
    Generates plane wavefront timings
    '''
    theta, phi = params
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp=np.sin(phi)
    K = np.array([-st*cp,-st*sp,-ct])
    dX = Xants - np.array([0.,0.,groundAltitude])
    tants = np.dot(dX,K) / cr 
 
    return (tants)

#@njit(**kwd)
def SWF_model(params, Xants, verbose=False, cr=1.0):
    '''
    Computes predicted wavefront timings for the spherical case.
    Inputs: params = theta, phi, r_xmax, t_s
    \theta, \phi are the spherical angular coordinates of Xmax, and  
    r_xmax is the distance of Xmax to the reference point of coordinates (0, 0, groundAltitude)
    c_r is the speed of light in vacuum, in units of c_light
    '''
    theta, phi, r_xmax, t_s = params
    nants = Xants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([-st*cp, -st*sp, -ct])
    Xmax = -r_xmax * K + np.array([0., 0., groundAltitude])
    tants = np.zeros(nants)
    for i in range(nants):
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i, :])
        #n_average = 1
        dX = Xants[i, :] - Xmax
        tants[i] = t_s + n_average / cr * np.linalg.norm(dX)

    return (tants)

def SWF_model_iminuit(XantsT,params):
    return SWF_model(params,XantsT.T)

@njit(**kwd,parallel=False)
def SWF_loss(params, Xants, tants, verbose=False, log = False, cr=1.0):

    '''
    Defines Chi2 by summing model residuals over antennas  (i):
    loss = \sum_i ( cr(tants[i]-t_s) - \sqrt{(Xants[i,0]-x_s)**2)+(Xants[i,1]-y_s)**2+(Xants[i,2]-z_s)**2} )**2
    where:
    Xants are the antenna positions (shape=(nants,3))
    tants are the trigger times (shape=(nants,))
    x_s = \sin(\theta)\cos(\phi)
    y_s = \sin(\theta)\sin(\phi)
    z_s = \cos(\theta)

    Inputs: params = theta, phi, r_xmax, t_s
    \theta, \phi are the spherical coordinates of the vector K
    t_s is the source emission time
    cr is the radiation speed in medium, by default 1 since time is expressed in m.
    '''


    if (log is True):
        # Pass r_xmax+t_s, np.log10(r_xmax - t_s) instead of r_xmax, t_s
        theta, phi, sm, logdf = params
        df = 10.**logdf
        r_xmax = (df+sm)/2.
        t_s    = (-df+sm)/2.
    else:
        theta, phi, r_xmax, t_s = params
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([-st*cp,-st*sp,-ct])
    Xmax = -r_xmax * K + np.array([0.,0.,groundAltitude]) # Xmax is in the opposite direction to shower propagation.
    #Xmax = -r_xmax * K + Xcore #Xcore is chosen taking the mean postion of triggered antennas

    # Make sure Xants and tants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible",tants.shape, Xants.shape)
        return None
    tmp = 0.
    for i in range(nants):
        # Compute average refraction index between emission and observer
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        #n_average = 1
        #if (verbose) :
        #    print('n_average = ',n_average)
        ## n_average = 1.0 #DEBUG
        dX = Xants[i,:] - Xmax
        # Spherical wave front
        res = cr*(tants[i]-t_s) - n_average*np.linalg.norm(dX)
        tmp += res*res

    chi2 = tmp
    if (verbose):
        print("theta,phi,r_xmax,t_s = ",theta,phi,r_xmax,t_s)
        print ("Chi2 = ",chi2)
    return(chi2)

@njit(**kwd)
def log_SWF_loss(params, Xants, tants, verbose=False, cr=1.0):
    return np.log10(SWF_loss(params,Xants,tants,verbose=verbose,cr=1.0))


@njit(**kwd)
def SWF_grad(params, Xants, tants, verbose=False, cr=1.0):
    '''
    Gradient of SWF_loss, w.r.t. theta, phi, r_xmax and t_s
    Note that this gradient is approximate in the sense that it 
    does not take into account the variations of the line of sight
    mean refractive index with Xmax(theta,phi,r_xmax)
    '''
    theta, phi, r_xmax, t_s = params
    # print("theta,phi,r_xmax,t_s = ",theta,phi,r_xmax,t_s)
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([-st*cp,-st*sp,-ct])
    Xmax = -r_xmax * K + np.array([0.,0.,groundAltitude]) # Xmax is in the opposite direction to shower propagation.
    # Derivatives of Xmax, w.r.t. theta, phi, r_xmax
    dK_dtheta = np.array([ct*cp,ct*sp,-st])
    dK_dphi   = np.array([-st*sp,st*cp,0.])
    dXmax_dtheta = -r_xmax*dK_dtheta
    dXmax_dphi   = -r_xmax*dK_dphi
    dXmax_drxmax = -K
    
    jac = np.zeros(4)
    for i in range(nants):
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        ## n_average = 1.0 ## DEBUG
        dX = Xants[i,:] - Xmax
        ndX = np.linalg.norm(dX)
        res = cr*(tants[i]-t_s) - n_average*ndX
        # Derivatives w.r.t. theta, phi, r_xmax, t_s
        jac[0] += -2*n_average*np.dot(-dXmax_dtheta,dX)/ndX * res
        jac[1] += -2*n_average*np.dot(-dXmax_dphi,  dX)/ndX * res
        jac[2] += -2*n_average*np.dot(-dXmax_drxmax,dX)/ndX * res
        jac[3] += -2*cr                                     * res 
    if (verbose):
        print ("Jacobian = ",jac)
    return(jac)


@njit(**kwd)
def SWF_hess(params, Xants, tants, verbose=False, cr=1.0):
    '''
    Hessian of SWF loss, w.r.t. theta, phi, r_xmax, t_s
    '''
    theta, phi, r_xmax, t_s = params
    # print("theta,phi,r_xmax,t_s = ",theta,phi,r_xmax,t_s)
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([-st*cp,-st*sp,-ct])
    Xmax = -r_xmax * K + np.array([0.,0.,groundAltitude]) # Xmax is in the opposite direction to shower propagation.
    # Derivatives of Xmax, w.r.t. theta, phi, r_xmax
    dK_dtheta = np.array([ct*cp,ct*sp,-st])
    dK_dphi   = np.array([-st*sp,st*cp,0.])
    dXmax_dtheta = -r_xmax*dK_dtheta
    dXmax_dphi   = -r_xmax*dK_dphi
    dXmax_drxmax = -K
    ### TO BE WRITTEN... WORTH IT ?

@njit(**kwd)
def SWF_residuals(params, Xants, tants, verbose=False, cr=1.0):

    '''
    Computes timing residuals for each antenna (i):
    residual[i] = ( cr(tants[i]-t_s) - \sqrt{(Xants[i,0]-x_s)**2)+(Xants[i,1]-y_s)**2+(Xants[i,2]-z_s)**2} )**2
    where:
    Xants are the antenna positions (shape=(nants,3))
    tants are the trigger times (shape=(nants,))
    x_s = \sin(\theta)\cos(\phi)
    y_s = \sin(\theta)\sin(\phi)
    z_s = \cos(\theta)

    Inputs: params = theta, phi, r_xmax, t_s
    \theta, \phi are the spherical coordinates of the vector K
    t_s is the source emission time
    cr is the radiation speed in medium, by default 1 since time is expressed in m.
    '''

    theta, phi, r_xmax, t_s = params
    # print("theta,phi,r_xmax,t_s = ",theta,phi,r_xmax,t_s)
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([-st*cp,-st*sp,-ct])
    Xmax = -r_xmax * K + np.array([0.,0.,groundAltitude]) # Xmax is in the opposite direction to shower propagation.

    # Make sure Xants and tants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible",tants.shape, Xants.shape)
        return None
    tmp = 0.
    res = np.zeros(nants)
    for i in range(nants):
        # Compute average refraction index between emission and observer
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        ## n_average = 1.0 #DEBUG
        dX = Xants[i,:] - Xmax
        # Spherical wave front
        res[i] = cr*(tants[i]-t_s) - n_average*np.linalg.norm(dX)

    return(res)

@njit(**kwd)
def SWF_simulation(params, Xants, sigma_t = 5e-9, iseed=1234, cr=1.0):
    '''
    Computes simulated wavefront timings for the spherical case.
    Inputs: params = theta, phi, r_xmax, t_s
    \theta, \phi are the spherical angular coordinates of Xmax, and  
    r_xmax is the distance of Xmax to the reference point of coordinates (0,0,groundAltitude)
    sigma_t is the timing jitter noise, in ns
    iseed is the integer random seed of the noise generator
    c_r is the speed of light in vacuum, in units of c_light
    '''
    theta, phi, r_xmax, t_s = params
    nants = Xants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([-st*cp, -st*sp, -ct])
    Xmax = -r_xmax * K + np.array([0.,0.,groundAltitude])
    tants = np.zeros(nants)
    for i in range(nants):
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        dX = Xants[i,:] - Xmax
        tants[i] = t_s + n_average / cr * np.linalg.norm(dX)

    np.random.seed(iseed)
    n = np.random.standard_normal(tants.size) * sigma_t * c_light
    return (tants + n)

@njit(**kwd)
def ADF_3D_parameters(params, Aants, Xants, Xmax, asym_coeff=0.01):
    
    '''

    Computes amplitude prediction for each antenna (i):
    residuals[i] = f_i^{ADF}(\theta,\phi,\delta\omega,A,r_xmax)
    where the ADF function reads:
    
    f_i = f_i(\omega_i, \eta_i, \alpha, l_i, \delta_omega, A)
        = A/l_i f_geom(\alpha, \eta_i) f_Cerenkov(\omega,\delta_\omega)
    
    where 
    
    f_geom(\alpha, \eta_i) = (1 + B \sin(\alpha))**2 \cos(\eta_i) # B is here the geomagnetic asymmetry
    f_Cerenkov(\omega_i,\delta_\omega) = 1 / (1+4{ (\tan(\omega_i)/\tan(\omega_c))**2 - 1 ) / \delta_\omega }**2 )
    
    Input parameters are: params = theta, phi, delta_omega, amplitude
    \theta, \phi define the shower direction angles, \delta_\omega the width of the Cerenkov ring, 
    A is the amplitude paramater, r_xmax is the norm of the position vector at Xmax.

    Derived parameters are: 
    \alpha, angle between the shower axis and the magnetic field
    \eta_i is the azimuthal angle of the (projection of the) antenna position in shower plane
    \omega_i is the angle between the shower axis and the vector going from Xmax to the antenna position

    '''

    theta, phi, delta_omega, amplitude = params
    nants = Xants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    # Define shower basis vectors
    K = np.array([-st*cp,-st*sp,-ct])
    K_plan = np.array([K[0],K[1]])
    KxB = np.cross(K,Bvec); KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K,KxB); KxKxB /= np.linalg.norm(KxKxB)
    # Coordinate transform matrix
    mat = np.vstack((KxB,KxKxB,K))
    # 
    XmaxDist = (groundAltitude-Xmax[2])/K[2]
    # print('XmaxDist = ',XmaxDist)

    theta_deg = np.rad2deg(theta)
    asym_coeff = -0.003*theta_deg+0.220
    asym = asym_coeff/np.sqrt(1. - np.dot(K,Bvec)**2)
    #

    # Loop on antennas. Here no precomputation table is possible for Cerenkov angle computation.
    # Calculation needs to be done for each antenna.
    res = np.zeros(nants)
    eta_array = np.zeros(nants)
    omega_array = np.zeros(nants)
    omega_cr_analytic_array = np.zeros(nants)
    omega_cr_analytic_effectif_array = np.zeros(nants)
    omega_cerenkov_simu_array = np.zeros(nants)
    Xb = Xmax - 2.0e3*K
    Xa = Xmax + 2.0e3*K
    for i in range(nants):
        # Antenna position from Xmax
        dX = Xants[i,:]-Xmax
        # Expressed in shower frame coordinates
        dX_sp = np.dot(mat,dX)
        #
        l_ant = np.linalg.norm(dX)
        eta = np.arctan2(dX_sp[1],dX_sp[0])
        omega = np.arccos(np.dot(K,dX)/l_ant)

        omega_cr = compute_Cerenkov_3D(Xants[i,:],K,XmaxDist,Xmax,2.0e3,groundAltitude)
        omega_cr_analytic = np.arccos(1./RefractionIndexAtPosition(Xmax))
        omega_cr_analytic_effectif = np.arccos(1./ZHSEffectiveRefractionIndex(Xmax, np.array([0, 0, groundAltitude])))
        # print ("omega_cr = ",omega_cr)

        theta_deg = np.rad2deg(theta) 
        if theta_deg < 70: 
            omega_cr = min(omega_cr, np.deg2rad(0.6))

        # Distribution width. Here rescaled by ratio of cosines (why ?)
        width = ct / (dX[2]/l_ant) * delta_omega
        # Distribution
        adf = amplitude/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/delta_omega )**2)
        adf *= 1. + asym*np.cos(eta) # 
        # Chi2
        res[i]= (Aants[i]-adf)
        eta_array[i] = eta
        omega_array[i] = omega
        omega_cr_analytic_array[i] = omega_cr_analytic
        omega_cr_analytic_effectif_array[i] = omega_cr_analytic_effectif
        omega_cerenkov_simu_array[i]= omega_cr

    return(eta_array, omega_array, omega_cerenkov_simu_array, omega_cr_analytic_array, omega_cr_analytic_effectif_array)  

@njit(**kwd)
def ADF_3D_loss(params, Aants, Xants, Xmax, asym_coeff=0.01, verbose=False):
    
    '''

    Computes amplitude prediction for each antenna (i):
    residuals[i] = f_i^{ADF}(\theta,\phi,\delta\omega,A,r_xmax)
    where the ADF function reads:
    
    f_i = f_i(\omega_i, \eta_i, \alpha, l_i, \delta_omega, A)
        = A/l_i f_geom(\alpha, \eta_i) f_Cerenkov(\omega,\delta_\omega)
    
    where 
    
    f_geom(\alpha, \eta_i) = (1 + B \sin(\alpha))**2 \cos(\eta_i) # B is here the geomagnetic asymmetry
    f_Cerenkov(\omega_i,\delta_\omega) = 1 / (1+4{ (\tan(\omega_i)/\tan(\omega_c))**2 - 1 ) / \delta_\omega }**2 )
    
    Input parameters are: params = theta, phi, delta_omega, amplitude
    \theta, \phi define the shower direction angles, \delta_\omega the width of the Cerenkov ring, 
    A is the amplitude paramater, r_xmax is the norm of the position vector at Xmax.

    Derived parameters are: 
    \alpha, angle between the shower axis and the magnetic field
    \eta_i is the azimuthal angle of the (projection of the) antenna position in shower plane
    \omega_i is the angle between the shower axis and the vector going from Xmax to the antenna position

    '''

    theta, phi, delta_omega, amplitude = params
    nants = Xants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    # Define shower basis vectors
    K = np.array([-st*cp,-st*sp,-ct])
    K_plan = np.array([K[0],K[1]])
    KxB = np.cross(K,Bvec); KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K,KxB); KxKxB /= np.linalg.norm(KxKxB)
   
    # Coordinate transform matrix
    mat = np.vstack((KxB,KxKxB,K))
    # 
    XmaxDist = (groundAltitude-Xmax[2])/K[2]

    # print('XmaxDist = ',XmaxDist)
    theta_deg = np.rad2deg(theta)
    asym_coeff = -0.003*theta_deg+0.220
    asym = asym_coeff/np.sqrt(1. - np.dot(K,Bvec)**2)
    #

    # Loop on antennas. Here no precomputation table is possible for Cerenkov angle computation.
    # Calculation needs to be done for each antenna.
    tmp = 0.
    res = np.zeros(nants)
    for i in range(nants):
        # Antenna position from Xmax
        dX = Xants[i,:]-Xmax
        # Expressed in shower frame coordinates
        dX_sp = np.dot(mat,dX)
        #
        l_ant = np.linalg.norm(dX)
        eta = np.arctan2(dX_sp[1],dX_sp[0])
        omega = np.arccos(np.dot(K,dX)/l_ant)
        
        omega_cr = compute_Cerenkov_3D(Xants[i,:],K,XmaxDist,Xmax,2.0e3,groundAltitude)
        #omega_cr= np.arccos(1./RefractionIndexAtPosition(Xmax))
    
        #print ("omega_cr = ",np.rad2deg(omega_cr))

        theta_deg = np.rad2deg(theta)
        if theta_deg <70: 
            #print('theta CR', 180-theta_deg)
            omega_cr = min(omega_cr, np.deg2rad(0.6))

        # Distribution width. Here rescaled by ratio of cosines (why ?)
        width = ct / (dX[2]/l_ant) * delta_omega
        #print('width', width)

        # Distribution
        adf = amplitude/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/delta_omega)**2)
        adf *= 1. + asym*np.cos(eta) # 
        # Chi2
        tmp += (Aants[i]-adf)**2
        #tmp += (Aants[i]-adf)**2/(0.075*Aants[i])**2 #divide by uncertainties (7.5% for DC2 simulations)
    chi2 = tmp
    if (verbose):
        print ("params = ",np.rad2deg(params[:2]),params[2:]," Chi2 = ",chi2)
    return(chi2)

#@njit(**kwd,nopython=True)
def ADF_3D_model(params, Xants, Xmax, asym_coeff=0.01):
    
    '''

    Computes amplitude prediction for each antenna (i):
    residuals[i] = f_i^{ADF}(\theta,\phi,\delta\omega,A,r_xmax)
    where the ADF function reads:
    
    f_i = f_i(\omega_i, \eta_i, \alpha, l_i, \delta_omega, A)
        = A/l_i f_geom(\alpha, \eta_i) f_Cerenkov(\omega,\delta_\omega)
    
    where 
    
    f_geom(\alpha, \eta_i) = (1 + B \sin(\alpha))**2 \cos(\eta_i) # B is here the geomagnetic asymmetry
    f_Cerenkov(\omega_i,\delta_\omega) = 1 / (1+4{ (\tan(\omega_i)/\tan(\omega_c))**2 - 1 ) / \delta_\omega }**2 )
    
    Input parameters are: params = theta, phi, delta_omega, amplitude
    \theta, \phi define the shower direction angles, \delta_\omega the width of the Cerenkov ring, 
    A is the amplitude paramater, r_xmax is the norm of the position vector at Xmax.

    Derived parameters are: 
    \alpha, angle between the shower axis and the magnetic field
    \eta_i is the azimuthal angle of the (projection of the) antenna position in shower plane
    \omega_i is the angle between the shower axis and the vector going from Xmax to the antenna position

    '''

    theta, phi, delta_omega, amplitude = params
    nants = Xants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    # Define shower basis vectors
    K = np.array([-st*cp,-st*sp,-ct])
    K_plan = np.array([K[0],K[1]])
    KxB = np.cross(K,Bvec); KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K,KxB); KxKxB /= np.linalg.norm(KxKxB)
    # Coordinate transform matrix
    mat = np.vstack((KxB,KxKxB,K))
    # 
    XmaxDist = (groundAltitude-Xmax[2])/K[2]
    # print('XmaxDist = ',XmaxDist)

    theta_deg = np.rad2deg(theta)
    asym_coeff = -0.003*theta_deg+0.220
    asym = asym_coeff/np.sqrt(1. - np.dot(K,Bvec)**2)

    # Loop on antennas. Here no precomputation table is possible for Cerenkov angle computation.
    # Calculation needs to be done for each antenna.
    res = np.zeros(nants)
    for i in range(nants):
        # Antenna position from Xmax
        dX = Xants[i,:]-Xmax
        # Expressed in shower frame coordinates
        dX_sp = np.dot(mat,dX)
        #
        l_ant = np.linalg.norm(dX)
        eta = np.arctan2(dX_sp[1],dX_sp[0])
        omega = np.arccos(np.dot(K,dX)/l_ant)

        omega_cr = compute_Cerenkov_3D(Xants[i,:],K,XmaxDist,Xmax,2.0e3,groundAltitude)
        # print ("omega_cr = ",omega_cr)

        theta_deg = np.rad2deg(theta) 
        if theta_deg <70: 
            #print('theta CR', 180-theta_deg)
            omega_cr = min(omega_cr, np.deg2rad(0.6))

        # Distribution width. Here rescaled by ratio of cosines (why ?)
        width = ct / (dX[2]/l_ant) * delta_omega
        # Distribution
        adf = amplitude/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/delta_omega )**2)
        adf *= 1. + asym*np.cos(eta) # 
        # Chi2
        res[i]= adf

    return(res)


@njit(**kwd)
def compute_alpha_3D(Xant, K, groundAltitude):
    dXcore = Xant - np.array([0.,0.,groundAltitude]) 
    U = dXcore / np.linalg.norm(dXcore)
    # Compute angle between shower direction and (horizontal) direction to observer
    alpha = np.arccos(np.dot(K,U))
    alpha = np.pi-alpha
    return (alpha)

@njit(**kwd)
def compute_U(Xant, groundAltitude):
    dXcore = Xant - np.array([0.,0.,groundAltitude]) 
    U = dXcore / np.linalg.norm(dXcore)
    return (U)

@njit(**kwd)
def compute_observer_position_3D(omega,Xmax,Xant,U,K,xmaxDist,alpha):
    '''
    Given angle omega between shower direction (K) and line joining Xmax and observer's position,
    Xmax position and Xant antenna position, and unit vector (U) to observer from shower core, compute
    coordinates of observer
    '''

    # Compute rotation axis. Make sure it is normalized
    Rot_axis = np.cross(U,K)
    Rot_axis /= np.linalg.norm(Rot_axis)
    # Compute rotation matrix from Rodrigues formula
    Rotmat = rotation(-omega,Rot_axis)
    # Define rotation using scipy's method
    # Rotation = R.from_rotvec(-omega * Rot_axis)
    # print('#####')
    # print(Rotation.as_matrix())
    # print('#####')
    # Dir_obs  = Rotation.apply(K)
    Dir_obs = np.dot(Rotmat,K)
    # Compute observer's position
    # this assumed coincidence was computed at antenna altitude)
    #t = (Xant[2] - Xmax[2])/Dir_obs[2]
    # This assumes coincidence is computed at fixed alpha, i.e. along U, starting from Xcore
    t = np.sin(alpha)/np.sin(alpha+omega) * xmaxDist
    X = Xmax + t*Dir_obs
    return (X)

    #def logprob(angles, coords, times):
#    return -0.5*PWF_loss(angles, coords, times)


@njit(**kwd)
def compute_delay_3D(omega,Xmax, Xa, Xb,Xant,U,K,alpha,delta,xmaxDist):

    X = compute_observer_position_3D(omega,Xmax,Xant,U,K,xmaxDist,alpha)
    # print('omega = ',omega,'X_obs = ',X)
    n2 = ZHSEffectiveRefractionIndex(Xa,X)
    # print('n0 = ',n0)
    n1 = ZHSEffectiveRefractionIndex(Xb, X)
    # print('n1 = ',n1)
    res = minor_equation(omega,n2,n1,alpha, delta, xmaxDist)
    # print('delay = ',res)
    return(res)

@njit(**kwd)
def compute_delay_3D_master_equation(omega, Xmax, Xa, Xb,Xant,U,K,alpha,delta,xmaxDist):

    X = compute_observer_position_3D(omega,Xmax,Xant,U,K,xmaxDist,alpha)
    # print('omega = ',omega,'X_obs = ',X)
    n2 = ZHSEffectiveRefractionIndex(Xa, X)
    # print('n0 = ',n0)
    n1 = ZHSEffectiveRefractionIndex(Xb, X)
    # print('n1 = ',n1)
    res = master_equation(omega, n2, n1, alpha, delta, xmaxDist)
    # print('delay = ',res)
    return(res)

@njit(**kwd)
def compute_Cerenkov_3D(Xant, K, xmaxDist, Xmax, delta, groundAltitude):

    '''
    Solve for Cerenkov angle by minimizing
    time delay between light rays originating from Xb and Xmax and arriving
    at observer's position. 
    Xant:  (single) antenna position 
    K:     direction vector of shower
    Xmax:  coordinates of Xmax point
    delta: distance between Xmax and Xb points
    groundAltitude: self explanatory

    Returns:     
    omega: angle between shower direction and line joining Xmax and observer's position

    '''

    # Compute coordinates of point before Xmax
    Xb = Xmax - delta*K
    # Compute coordinates of point after Xmax
    Xa = Xmax +delta*K

    #dXcore = Xant - np.array([0.,0.,groundAltitude])
    # Core of shower, taken at groundAltitude for reference
    # Ground altitude might be computed later as a derived quantity, e.g. 
    # as the median of antenna altitudes.
    Xcore = Xmax + xmaxDist * K 
    dXcore = Xant - Xcore

    # Direction vector to observer's position from shower core
    # This is a bit dangerous for antennas numerically close to shower core... 
    U = dXcore / np.linalg.norm(dXcore)
    # Compute angle between shower direction and (horizontal) direction to observer
    alpha = np.arccos(np.dot(K,U))
    alpha = np.pi-alpha


    # Now solve for omega
    # Starting point at standard value acos(1/n(Xmax)) 
    omega_cr_guess = np.arccos(1./RefractionIndexAtPosition(Xmax))
    # print("###############")
    # omega_cr = fsolve(compute_delay,[omega_cr_guess])
    omega_cr = newton(compute_delay_3D, omega_cr_guess, args=(Xmax, Xa, Xb, Xant,U,K,alpha,delta, xmaxDist),verbose=False)
    ### DEBUG ###
    # omega_cr = omega_cr_guess
    return(omega_cr)

#@njit(**kwd)
def ADF_grad(params, Aants, Xants, Xmax, asym_coeff=0.01,verbose=False):
    
    theta, phi, delta_omega, amplitude = params
    nants = Aants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    # Define shower basis vectors
    K = np.array([-st*cp,-st*sp,-ct])
    K_plan = np.array([K[0],K[1]])
    KxB = np.cross(K,Bvec); KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K,KxB); KxKxB /= np.linalg.norm(KxKxB)
    # Coordinate transform matrix
    mat = np.vstack((KxB,KxKxB,K))
    # 
    XmaxDist = (groundAltitude-Xmax[2])/K[2]
    # print('XmaxDist = ',XmaxDist)
    asym = asym_coeff * (1. - np.dot(K,Bvec)**2) # Azimuthal dependence, in \sin^2(\alpha)
    #
    # Make sure Xants and tants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of Aants and Xants are incompatible",Aants.shape, Xants.shape)
        return None

    # Precompute an array of Cerenkov angles to interpolate over (as in Valentin's code)
    omega_cerenkov = np.zeros(2*n_omega_cr+1)
    xi_table = np.arange(2*n_omega_cr+1)/n_omega_cr*np.pi
    for i in range(n_omega_cr+1):
        omega_cerenkov[i] = compute_Cerenkov_3D(xi_table[i],K,XmaxDist,Xmax,2.0e3,groundAltitude)
    # Enforce symmetry
    omega_cerenkov[n_omega_cr+1:] = (omega_cerenkov[:n_omega_cr])[::-1]

    #Output antenas amplitudes for comparisons
    Aants_out = np.zeros((nants, 2))
    # Loop on antennas
    jac = np.zeros(4)
    for i in range(nants):
        # Antenna position from Xmax
        dX = Xants[i,:]-Xmax
        # Expressed in shower frame coordinates
        dX_sp = np.dot(mat,dX)
        #
        l_ant = np.linalg.norm(dX)
        eta = np.arctan2(dX_sp[1],dX_sp[0])
        omega = np.arccos(np.dot(K,dX)/l_ant)

        omega_cr = compute_Cerenkov_3D(Xants[i,:],K,XmaxDist,Xmax,2.0e3,groundAltitude)
        width = ct / (dX[2]/l_ant) * delta_omega
        # Distribution
        f_cerenkov = 1. / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/width )**2)
        adf = amplitude/l_ant * f_cerenkov
        adf *= 1. + asym*np.cos(eta) #
        res = Aants[i] - adf
        #
        dK_dtheta = np.array([ct*cp, ct*sp,-st])
        dfgeom_dtheta = -2.*asym_coeff*np.cos(eta)*(np.dot(K,Bvec))*(np.dot(dK_dtheta,Bvec))
        dres_dtheta = (-amplitude/l_ant)*f_cerenkov*dfgeom_dtheta
        #
        dK_dphi = np.array([-st*sp, st*cp, 0.])
        dfgeom_dphi = -2.*asym_coeff*np.cos(eta)*(np.dot(K,Bvec))*(np.dot(dK_dphi,Bvec))
        dres_dphi = (-amplitude/l_ant)*f_cerenkov*dfgeom_dphi
        #
        term1 = (np.tan(omega)/np.tan(omega_cr))**2 - 1. 
        dfcerenkov_ddelta_omega = (8.*l_ant**2.*(1/ct**2.)*term1**2.)/(delta_omega**3.*dX[2]**2.*(1+(4.*l_ant**2.*(1/ct**2.)*term1**2.)/(delta_omega**2.*dX[2]**2.))**2.)
        dres_ddelta_omega =  (-amplitude/l_ant)*(1. + asym*np.cos(eta))*dfcerenkov_ddelta_omega
        #
        dres_damplitude = (-1./l_ant) * f_cerenkov * (1. + asym*np.cos(eta))
        # grad
        jac[0] += 2.*res*dres_dtheta
        jac[1] += 2.*res*dres_dphi
        jac[2] += 2.*res*dres_ddelta_omega
        jac[3] += 2.*res*dres_damplitude

        return(jac)

