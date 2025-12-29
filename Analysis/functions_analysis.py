import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

"""
Physics and analysis functions for cosmic-ray reconstruction.

This module contains geometry utilities, wavefront models, refraction index
computation and background fitting tools used in the ADF candidate analysis.

All angles are in degrees unless stated otherwise.
All distances are in meters.
All times are in nanoseconds.
"""

# =============================================================================
# Physical constants
# =============================================================================
c_light = 2.997924580e8
R_earth = 6371007.0

# Refractivity model parameters (ZHS)
ns = 325
kr = -0.1218

# =============================================================================
# Geometry utilities
# =============================================================================

def compute_omega(theta, phi, Xants, Xmax):
    """
    Compute the opening angle omega between antenna directions and shower axis.

    Parameters
    ----------
    theta, phi : float
        Shower direction angles [deg].
    Xants : ndarray, shape (N, 3)
        Antenna positions [m].
    Xmax : ndarray, shape (3,)
        Shower maximum position [m].

    Returns
    -------
    omega : ndarray, shape (N,)
        Opening angle for each antenna [deg].
    """
    K = compute_shower_axis(theta, phi)     

    dX = Xants - Xmax                       
    l_ant = np.linalg.norm(dX, axis=1)      
    cos_omega = (dX @ K) / l_ant         
    omega = np.arccos(cos_omega)             

    return np.rad2deg(omega)

def compute_eta(theta, phi, Xants, Xmax, Bn):
    """
    Compute the azimutal angle eta in the shower plane.

    Parameters
    ----------
    theta, phi : float
        Shower direction angles [deg].
    Xants : ndarray, shape (N, 3)
        Antenna positions [m].
    Xmax : ndarray, shape (3,)
        Shower maximum position [m].

    Returns
    -------
    omega : ndarray, shape (N,)
        Opening angle for each antenna [deg].
    """
    k = compute_shower_axis(theta, phi) 
    kxB = np.cross(k, Bn); 
    kxB /= np.linalg.norm(kxB)
    kxkxB = np.cross(k,kxB); 
    kxkxB /= np.linalg.norm(kxkxB)    
    mat = np.vstack((kxB,kxkxB,k))
    dX = Xants - Xmax
    dX_sp = mat @ dX.T
    eta = np.arctan2(dX_sp[1],dX_sp[0])           

    return np.rad2deg(eta)


def compute_shower_axis(theta, phi):
    """
    Compute the shower propagation unit vector.

    Parameters
    ----------
    theta : float
        Zenith angle [deg].
    phi : float
        Azimuth angle [deg].

    Returns
    -------
    k : ndarray, shape (3,)
        Shower propagation unit vector.
    """
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    # vecteur de propagation
    K = np.array([-st*cp, -st*sp, -ct])      # shape (3,)
    return K

def generate_cone_surface_vectors(k, omega, n=10):
    """
    Generate vectors uniformly distributed on a cone surface.

    Parameters
    ----------
    k : ndarray, shape (3,)
        Cone axis unit vector.
    omega : float
        Cone opening angle [deg].
    n : int, optional
        Number of vectors (default: 10).

    Returns
    -------
    vectors : ndarray, shape (n, 3)
        Unit vectors lying on the cone surface.
    """

    # Generate n uniform angles around the axis (azimuthal angle)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    omega = omega*np.pi/180

    # Spherical to Cartesian conversion with fixed polar angle omega
    x = np.sin(omega) * np.cos(theta)
    y = np.sin(omega) * np.sin(theta)
    z = np.full_like(x, np.cos(omega))  # fixed z for all points (cos(omega))

    # These are vectors on a cone aligned with the z-axis
    vectors = np.vstack((x, y, z)).T  # shape (n, 3)

    # We now compute the rotation matrix that aligns z-axis to vector k
    z_axis = np.array([0, 0, 1])
    if np.allclose(k, z_axis):
        rot_matrix = np.eye(3)  # No rotation needed
    elif np.allclose(k, -z_axis):
        # Special case: 180° rotation around any perpendicular axis
        rot_matrix = np.array([[-1, 0, 0],
                               [0, -1, 0],
                               [0,  0, 1]])
    else:
        # Use Rodrigues' rotation formula to get rotation matrix
        v = np.cross(z_axis, k)
        c = np.dot(z_axis, k)
        s = np.linalg.norm(v)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        rot_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)

    # Rotate all vectors from z-aligned cone to k-aligned cone
    rotated_vectors = vectors @ rot_matrix.T
    return rotated_vectors

def compute_core(k,xs,refAlt,disp=False):
    """
    Compute the shower core position at ground altitude.

    Parameters
    ----------
    k : ndarray, shape (3,)
        Shower direction unit vector.
    xs : ndarray, shape (3,)
        Emission point coordinates [m].
    disp : bool, optional
        If True, display diagnostic plots.

    Returns
    -------
    xc : ndarray, shape (3,)
        Core position at ground [m].
    """
    k = np.asarray(k).reshape(3,)    
    xs = np.asarray(xs).reshape(3,) 
    u = np.linspace(0, np.linalg.norm(xs*1.5), 5001)  # Distance from x0 (meters)
    traj = np.zeros((3,len(u)))
    traj[0,:] = k[0] * u + xs[0]
    traj[1,:] = k[1] * u + xs[1]
    traj[2,:] = k[2] * u + xs[2]
    
    z_sol = refAlt
    u_core = (z_sol - xs[2]) / k[2]
    xc = xs + k * u_core

    if disp:
        plt.figure()  
        plt.plot([xs[0], xc[0]], [xs[2], xc[2]], 'k')  # ligne trajectoire
        plt.plot(xc[0], xc[2], 'ro')                  # core
        plt.plot(xs[0], xs[2], 'ro')                 #emission point 
        plt.axhline(y=refAlt, color='olive', lw=2)     # sol
        plt.xlabel("x")
        plt.ylabel("z")
        plt.show()

        plt.figure(12)
        plt.plot([xs[1], xc[1]], [xs[2], xc[2]], 'k')  # ligne trajectoire
        plt.plot(xc[1], xc[2], 'ro')     # core
        plt.plot(xs[1], xs[2], 'ro')     #emission point        
        plt.axhline(y=refAlt, color='olive', lw=2)     # sol
        plt.xlabel("y")
        plt.ylabel("z")
        plt.show()

    return(xc)

def ZHSEffectiveRefractionIndex(X0,Xa):
    """
    Compute the effective refractive index between emission point and antenna.

    Parameters
    ----------
    X0 : ndarray, shape (3,)
        Emission point coordinates [m].
    Xa : ndarray, shape (3,)
        Antenna position [m].

    Returns
    -------
    n_eff : float
        Effective refractive index.
    """

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

# =============================================================================
# Wavefront and background models
# =============================================================================

def PWF_model(params, Xants, groundAltitude, c_light): 
    """
    Plane wavefront timing model.

    Parameters
    ----------
    params : tuple (theta, phi)
        Shower direction angles [rad].
    Xants : ndarray, shape (N, 3)
        Antenna positions [m].

    Returns
    -------
    tants : ndarray, shape (N,)
        Arrival times [ns].
    """
    theta, phi = params 
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp=np.sin(phi) 
    K = np.array([-st*cp,-st*sp,-ct]) 
    dX = Xants - np.array([0.,0.,groundAltitude]) 
    tants = np.dot(dX,K) / c_light *1e9 #ns 
    return (tants)

def SWF_model(params, Xants, groundAltitude, c_light):
    """
    Spherical wavefront timing model with refraction.

    Parameters
    ----------
    params : tuple (theta, phi, r_xmax, t_s)
        Shower and timing parameters.
    Xants : ndarray, shape (N, 3)
        Antenna positions [m].

    Returns
    -------
    tants : ndarray, shape (N,)
        Arrival times [ns].
    """
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
        tants[i] = t_s + n_average / c_light * np.linalg.norm(dX) *1e9 #ns

    return (tants)

def adf_fun(l,A, wc, dw):
    w = np.linspace(0,3,200)*np.pi/180
    wc = wc*np.pi/180
    dw = dw
    eta = 0
    f_geom = 1  # Ignore for now
    f_ch = 1/(1.+4.*( ( (np.tan(w)/np.tan(wc) )**2 -1. ) /dw )**2)
    f_adf = A/l*f_geom*f_ch
    return(w*180/np.pi,f_adf)

def model(l, kappa):
    """Background amplitude model A = kappa / l."""
    return kappa / l

def background_fit(A, l):
    """
    Fit background model to amplitudes.

    Parameters
    ----------
    A : ndarray
        Measured amplitudes.
    l : ndarray
        Distance between the emission point (Xsource) and the antenna [m].

    Returns
    -------
    y_fit : ndarray
        Fitted background.
    chi2_red : float
        Reduced chi-square of the fit.
    """
    popt, pcov = curve_fit(model, l, A)
    A_fit = popt[0]
    A_err = np.sqrt(pcov[0, 0])
    y_fit = model(l, A_fit)
    
    chi2 = np.sum(((A - y_fit) / (0.075 * A))**2)
    ndof = len(A) - 1  
    chi2_red = chi2 / ndof
    return y_fit, chi2_red

def compute_reduced_chi2(chi2_values, n_points, n_params=4):
    """
    Compute reduced chi-square.

    Parameters
    ----------
    chi2 : float or ndarray
        Chi-square value(s).
    n_points : int
        Number of data points.
    n_params : int, optional
        Number of fitted parameters.

    Returns
    -------
    chi2_red : float or ndarray
        Reduced chi-square.
    """
    dof = n_points - n_params
    reduced_chi2 = np.where(dof > 0, chi2_values / dof, np.nan)
    return reduced_chi2

# -----------------------------
# Geometry along Earth curvature
# -----------------------------

def get_local_zenith(zenith, local_height, start_height):
    """
    Compute local zenith angle at a given altitude accounting for Earth's curvature.

    Parameters
    ----------
    zenith : float
        Zenith angle at the starting height [deg].
    local_height : float
        Height at which to compute the local zenith [m].
    start_height : float
        Starting height [m]. = altitude = shower core height

    Returns
    -------
    zenith_at : float
        Zenith angle at local height [deg].
    """
    zenith_rad = np.deg2rad(zenith)
    delta = (R_earth + start_height)**2 * np.cos(zenith_rad)**2 \
            + (local_height - start_height)*(local_height + start_height + 2*R_earth)
    path_length = (R_earth + start_height)*np.cos(zenith_rad) + np.sqrt(delta)

    cos_theta = (path_length**2 + (R_earth + local_height)**2 - (R_earth + start_height)**2) \
                / (2 * path_length * (R_earth + local_height))
    zenith_at = (np.pi - np.arccos(cos_theta)) * 180./np.pi
    return zenith_at


def get_local_height(zenith, start_height, path_length):
    """
    Compute height at a given path length along a zenith direction.

    Parameters
    ----------
    zenith : float
        Zenith angle at the starting height [deg].
    start_height : float
        Starting height [m]. = altitude = shower core height
    path_length : float
        Distance along the shower path [m].

    Returns
    -------
    height_at : float
        Height at the path length [m].
    """
    zenith_rad = np.deg2rad(zenith)
    height_at = -R_earth + np.sqrt((R_earth + start_height)**2 + path_length**2
                                   - 2*path_length*(R_earth + start_height)*np.cos(zenith_rad))
    return height_at


# -----------------------------
# Atmospheric density models
# -----------------------------

def get_density(height, model='linsley'):
    """
    Return atmospheric density at a given height.

    Parameters
    ----------
    height : float
        Height above sea level [m].
    model : str
        Atmospheric model: 'isothermal', 'linsley', or 'chao'.

    Returns
    -------
    rho : float
        Atmospheric density [kg/m^3].
    """
    if model == 'isothermal':
        rho0 = 1.225
        M = 0.028966
        g = 9.81
        T = 288.
        R = 8.32
        rho = rho0 * np.exp(-g * M * height / (R * T))

    elif model == 'linsley':
        bl = np.array([12220., 11440., 13055.948, 5401.778, 10])
        cl = np.array([9941.86, 8781.53, 6361.43, 7721.70, 1e7])
        hl = np.array([4, 10, 40, 100, 113])*1e3
        if height >= hl[-1]:
            rho = 0
        else:
            idx = np.where((height >= hl[:-1]) & (height < hl[1:]))[0][0]
            rho = bl[idx]/cl[idx] * np.exp(-height/cl[idx])

    elif model == 'chao':
        bl = np.array([12586., 11701.8, 12289.6, 12288.9, 10])
        cl = np.array([10653.3, 9494.97, 6962.43, 6962.57, 1e7])
        hl = np.array([3.689, 9.378, 26.299, 100., 113])*1e3
        if height >= hl[-1]:
            rho = 0
        else:
            idx = np.where((height >= hl[:-1]) & (height < hl[1:]))[0][0]
            rho = bl[idx]/cl[idx] * np.exp(-height/cl[idx])
    else:
        raise ValueError("Model must be 'isothermal', 'linsley' or 'chao'.")

    return rho


# -----------------------------
# Grammage calculation
# -----------------------------

def compute_distance_grammage(zenith, xmax_distance, longitudinal_distance, shower_core_height, n_steps=1000):
    """
    Computes atmospheric grammage along shower path.

    Parameters
    ----------
    zenith : float
        Zenith angle at starting height [deg].
    xmax_distance : float
        Distance to shower maximum [m].
    longitudinal_distance : float
        Longitudinal distance along shower from injection point [m].
    shower_core_height : float
        Height of shower core [m].
    n_steps : int
        Number of integration steps.

    Returns
    -------
    X : float
        Integrated grammage [g/cm^2].
    """
    dl = longitudinal_distance / n_steps
    conversion_factor = 0.1  # kg/m² -> g/cm²

    X = 0.
    height = get_local_height(zenith, shower_core_height, xmax_distance)
    local_zenith = get_local_zenith(zenith, height, shower_core_height)

    for _ in range(n_steps):
        height_new = get_local_height(local_zenith, height, dl)
        zenith_new = get_local_zenith(local_zenith, height_new, height)

        if height_new < 0:
            break

        height_mid = 0.5*(height + height_new)
        dX = get_density(height_mid, 'linsley') * dl * conversion_factor
        X += dX

        height = height_new
        local_zenith = zenith_new

    return X

def compute_injection_point(azimuth, zenith, injection_height, shower_core_height):
    """
    Compute the injection point of the shower along its axis.
    
    Parameters
    ----------
    azimuth : float
        Azimuth angle [deg]
    zenith : float
        Zenith angle [deg]
    injection_height : float
        Height of injection [m]
    shower_core_height : float
        Shower core height [m]
    
    Returns
    -------
    np.ndarray
        Injection point coordinates [x, y, z] in meters
    """
    theta = np.deg2rad(zenith)
    phi = np.deg2rad(azimuth)
    
    # Shower direction unit vector
    k_shower = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    #print(k_shower)
    
    # Distance along shower axis considering Earth's curvature
    # I added np.pi - theta
    delta = ((R_earth + shower_core_height)**2 * np.cos(theta)**2 +
             (injection_height - shower_core_height) * (injection_height + shower_core_height + 2 * R_earth))
    injection_length = (R_earth + shower_core_height) * np.cos(theta) + np.sqrt(delta)
    #print(injection_length)
    
    # Injection point coordinates
    injection_point = -k_shower * injection_length
    
    injection_point[2] += shower_core_height  # correct z coordinate
    #print(injection_point)
    return injection_point

def compute_longitudinal_distance(azimuth, zenith, injection_height, shower_core_height,
                                  x_source, y_source, z_source):
    """
    Compute the longitudinal distance from the injection point to the source.
    
    Works with scalars or arrays.
    """
    # Ensure arrays
    azimuth = np.atleast_1d(azimuth)
    zenith = np.atleast_1d(zenith)
    injection_height = np.atleast_1d(injection_height)
    shower_core_height = np.atleast_1d(shower_core_height)
    x_source = np.atleast_1d(x_source)
    y_source = np.atleast_1d(y_source)
    z_source = np.atleast_1d(z_source)
    
    distances = []
    for i in range(len(zenith)):
        injection_point = compute_injection_point(azimuth[i], zenith[i],
                                                  injection_height[i], shower_core_height[i])
        source_point = np.array([x_source[i], y_source[i], z_source[i]])
        distances.append(np.linalg.norm(source_point - injection_point))
    
    return distances if len(distances) > 1 else distances[0]

def compute_grammage(zenith, xmax_distance, shower_core_height, longitudinal_distance):
    """
    Compute the grammage along the shower path.
    
    Parameters
    ----------
    zenith : float or array
        Zenith angle(s) [deg]
    xmax_distance : float or array
        Distance from shower maximum to ground [m]
    shower_core_height : float or array
        Shower core height [m]
    injection_height : float or array
        Injection height [m]
    longitudinal_distance : float or array
        Longitudinal distance from injection point to source [m]

    Returns
    -------
    np.ndarray
        Grammage [g/cm²]
    """
    zenith = np.atleast_1d(zenith)
    xmax_distance = np.atleast_1d(xmax_distance)
    shower_core_height = np.atleast_1d(shower_core_height)
    longitudinal_distance = np.atleast_1d(longitudinal_distance)
    
    return np.array([
        compute_distance_grammage(z, x, l, h)
        for z, x, l, h in zip(zenith, xmax_distance, longitudinal_distance, shower_core_height)
    ])

def recons_grammage(rec_azimuth, rec_zenith, x_source, y_source, z_source,
                    injection_height, shower_core_height, x_core, y_core):
    """
    Reconstruct the grammage of a shower given its reconstructed direction
    and source coordinates.

    Parameters
    ----------
    rec_azimuth : float or array
        Reconstructed azimuth angle(s) [deg]
    rec_zenith : float or array
        Reconstructed zenith angle(s) [deg]
    x_source, y_source, z_source : float or array
        Source coordinates [m]
    injection_height : float or array
        Height of injection [m]
    shower_core_height : float or array
        Shower core height [m]

    Returns
    -------
    grammage_recons : float or np.ndarray
        Reconstructed grammage [g/cm²]
    longitudinal_distance : float or np.ndarray
        Longitudinal distance from injection point to source [m]
    """
    # Compute distance along shower axis from injection point to source

    rec_azimuth = (180 + rec_azimuth)%360
    rec_zenith = 180 - rec_zenith
    longitudinal_distance = compute_longitudinal_distance(
        rec_azimuth, rec_zenith, injection_height, shower_core_height,
        x_source, y_source, z_source
    )

    # Compute straight-line distance from source to shower core on the ground
    source_distance = np.sqrt((x_source-x_core)**2 + (y_source-y_core)**2 + (z_source - shower_core_height)**2)

    # Compute reconstructed grammage
    grammage_recons = compute_grammage(rec_zenith, source_distance,
                                       shower_core_height, longitudinal_distance)

    return grammage_recons, longitudinal_distance



def load_antenna_positions(file_path, shower_core_height=1231.0):
    """
    Load and preprocess antenna positions from a text file.

    Parameters
    ----------
    file_path : str
        Path to the antenna position file.
    shower_core_height : float
        Height to add to z-coordinate (default: 1231.0 m).
    excluded_ids : list or None
        List of antenna_IDs to exclude.
    swap_xy : bool
        Whether to swap x and y coordinates and flip y.

    Returns
    -------
    pd.DataFrame
        Preprocessed antenna positions.
    """
    column_names = ['antenna_ID', 'x', 'y', 'z']
    df = pd.read_csv(file_path, sep=r'\s+', names=column_names, header=None)


    df[['y','x']] = df[['x','y']].values
    df['y'] = -df['y']

    df['z'] += shower_core_height
    df['antenna_ID'] = df['antenna_ID'].astype(int)

    # Remove first antennas and excluded_ids
    df = df[~df['antenna_ID'].between(0, 15)]

    return df

