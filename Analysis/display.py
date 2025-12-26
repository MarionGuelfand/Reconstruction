"""
Event Footprint and ADF Fit Visualization

This script processes reconstruction files for a given event and
visualizes:
1. Voltage vs Omega (ADF fit) for triggered antennas
2. Background fit vs distance from source
3. Triggered antenna positions on the ground with footprint
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import functions_analysis as func
import config as conf
import cuts as cuts

# -------------------------------
# Ground meshgrid (for plotting)
# -------------------------------
distm = 5000  # distance in meters
xmin, xmax = -distm, distm
ymin, ymax = -distm, distm


# -------------------------------
# Load reconstruction data
# -------------------------------
f_swf = conf.root_paths[1] + 'Rec_sphere_wave_recons.txt'
f_pwf = conf.root_paths[1] + 'Rec_plane_wave_recons.txt'
swf = np.genfromtxt(f_swf, dtype=float, autostrip=True)
pwf = np.genfromtxt(f_pwf, dtype=float, autostrip=True)
swf = np.atleast_2d(swf)
pwf = np.atleast_2d(pwf)
idpwf_rec = pwf[:,0]; nants_pwf = pwf[:,1]; theta_rec = pwf[:,2]
phi_rec = pwf[:,4]; chi2_pwf = pwf[:,5]; nants_swf = swf[:,1]
chi2_swf = swf[:,2]; xs_rec = swf[:,4:7]; r_xmax = swf[:,7]
t_0 = swf[:,8]; theta_rec_swf = swf[:,9]; phi_rec_swf = swf[:,10]

f_adf = conf.root_paths[1] + 'Rec_adf_recons.txt'
adf = np.genfromtxt(f_adf, dtype=float, autostrip=True)
adf = np.atleast_2d(adf)
idadf_rec = adf[:,0]; nants_adf = adf[:,1]; thadf_rec = adf[:,2]
phadf_rec = adf[:,4]; chi2_adf = adf[:,6]; dwadf_rec = adf[:,8]
aadf_rec = adf[:,9]

f_adff = conf.root_paths[1] + 'Rec_adf_parameters.txt'
adff = np.genfromtxt(f_adff, dtype=float, autostrip=True)
adff = np.atleast_2d(adff)
id_rec = adff[:,0]; mult_rec = adff[:,1]; am_rec = adff[:,2]
af_rec = adff[:,3]; eta_rec = adff[:,4]; w_rec = adff[:,5]
wc_rec = adff[:,6]; wcs_rec = adff[:,7]; l_ant_distance = adff[:,8]
xant_rec = adff[:,9:12]


f_info_rootfile = conf.root_paths[1] + 'Info_rootfile.txt'
df_info_rootfile = pd.read_csv(f_info_rootfile, sep=r"\s+", header=None,
                     names=["fullID", "rootfile", "run_id", "last_number", "event"])

f_antennas = conf.root_paths[1] + 'Rec_coinctable.txt'
df_ant = pd.read_csv(f_antennas, sep=r"\s+", header=None,
                     names=["id", "id_coinc", "time", "amp_data", "event_ID", "run"])

# --- Load second file: coord_antennas_withID.txt ---
f_coinc = conf.root_paths[1] + 'coord_antennas_withID.txt'
df_coord = pd.read_csv(f_coinc, sep=r"\s+", header=None,
                       names=["id", "x_antenna", "y_antenna", "z_antenna", "antenna_ID", "event_ID", "run"])

# --- Merge (join) based on event_ID and run ---
merged = pd.merge(df_ant, df_coord, on=["id", "event_ID", "run"], suffixes=("_rec", "_coord"))
pd.set_option('display.max_columns', None)
#print(merged)

#for each CD root file: load event number, number of triggered antennas + antenna IDs
f_CD_all = conf.root_paths[1] +  'events_CD_all_events.txt'
df_cd = pd.read_csv(f_CD_all, sep="\t", header=None, engine='python')
df_cd.rename(columns={0: "fullID", 1: "n_antennas"}, inplace=True)

# -------------------------------
# Loop over unique events
# -------------------------------
for idx in np.unique(merged['id_coinc']): 
    # Create masks for selections
    mask_adf     = (id_rec == idx)
    mask_adfrec  = (idadf_rec == idx)
    mask_pwf     = (idpwf_rec == idx)
    mask_coinc = (merged["id_coinc"] == idx) 
    mask_info = (df_info_rootfile["fullID"] == idx) 

    l_sel = l_ant_distance[mask_adf]
    A_sel = am_rec[mask_adf]


    if np.isnan(l_sel).any() or np.isnan(A_sel).any() or np.isinf(l_sel).any() or np.isinf(A_sel).any() or (l_sel == 0).any():
        continue


    xs_sel = xs_rec[mask_pwf]
    if np.isnan(xs_sel).any():
        continue

    aadf_sel = aadf_rec[mask_adfrec]
    if np.isnan(aadf_sel).any():
        continue

    dwadf_sel = dwadf_rec[mask_adfrec]
    #if (dwadf_sel > 2.99999).any():
        #print(f"ID {idx} ignored : aadf_rec > 2.99999")
    #    continue

    # -------------------------------
    # Apply quality cuts
    # -------------------------------
    w_rec_sel = w_rec[mask_adf]
    if (w_rec_sel > cuts.cuts["w_rec_max"]).any():
        continue

    theta_sel = thadf_rec[mask_adfrec]
    if (theta_sel > cuts.cuts["theta_max"]).any():
        continue


    nants_sel = nants_adf[mask_adfrec]
    #if (nants_sel <10).any():
        #print(f"ID {idx} ignored : aadf_rec > 2.99999")
    #    continue

    reduced_chi2 = func.compute_reduced_chi2(chi2_adf[mask_adfrec], 
                                        nants_adf[mask_adfrec], n_params=4)[0]
    
    reduced_chi2_swf = func.compute_reduced_chi2(chi2_swf[mask_pwf], 
                                        nants_swf[mask_pwf], n_params=4)[0]
    
    reduced_chi2_pwf = func.compute_reduced_chi2(chi2_pwf[mask_pwf], 
                                        nants_pwf[mask_pwf], n_params=2)[0]

    
    
    if reduced_chi2_swf > cuts.cuts["reduced_chi2_swf_max"]:
        continue
    if reduced_chi2 > cuts.cuts["reduced_chi2_adf_max"]:
        continue
    if reduced_chi2_pwf > cuts.cuts["reduced_chi2_pwf_max"]:
        continue
    
    A_backgroundfit, chi2_red_backgroundfit = func.background_fit(A_sel, l_sel)


    ## Geometry 
    dX = xant_rec[mask_adf,:]-xs_rec[mask_pwf,:]  # Vectors DU-Xs
    l_ants = np.linalg.norm(dX,axis=1)  # Distance to source
    l = np.mean(l_ants)

    # Direction vector
    k = func.compute_shower_axis(thadf_rec[mask_adfrec][0], phadf_rec[mask_adfrec][0])
    eta = np.deg2rad(func.compute_eta(thadf_rec[mask_adfrec][0], phadf_rec[mask_adfrec][0], xant_rec[mask_adf,:], xs_rec[mask_pwf,:], cuts.Bn))
    omega = np.deg2rad(func.compute_omega(thadf_rec[mask_adfrec][0], phadf_rec[mask_adfrec][0], xant_rec[mask_adf,:], xs_rec[mask_pwf,:]))

    # --- Compute core position of the shower ---
    xs = xs_rec[mask_pwf,:][0]
    xc = func.compute_core(k, xs, cuts.refAlt) 
    print("Core position:",xc)

    # Compute distance between source and core (in km)
    dist = np.linalg.norm(xs - xc) / 1000 #en km

    # Compute ellipse
    wcm = np.mean(wc_rec[mask_adf])
    npts = 100
    v_cone = func.generate_cone_surface_vectors(k, wcm, npts)
    print("Computing footprint with opening angle w_c = ",wcm,"deg")
    x_ell = np.zeros((npts,3))
    for i, v in enumerate(v_cone):
        x_ell[i,:] = func.compute_core(v, xs, cuts.refAlt)

    # Now evaluate errors
    xse = xs
    xse[2] = xse[2]+100 # 100m higher    
    the = thadf_rec[mask_adfrec][0]*np.pi/180+0.1*np.pi/180
    phie = phadf_rec[mask_adfrec][0]*np.pi/180-0.1*np.pi/180
    ct = np.cos(the)
    st = np.sin(the)
    cp = np.cos(phie)
    sp = np.sin(phie)
    ke = np.array([-st*cp, -st*sp, -ct])
    xce = func.compute_core(ke, xse, cuts.refAlt)
    # Compute ellipse
    npts = 200
    v_cone = func.generate_cone_surface_vectors(ke, wcm, npts)
    x_elle = np.zeros((npts,3))
    for i, v in enumerate(v_cone):
        x_elle[i,:] = func.compute_core(v, xse, cuts.refAlt)
    xse2 = xs
    xse2[2] = xse2[2]-100 # 100m higher    
    the = thadf_rec[mask_adfrec][0]*np.pi/180-0.1*np.pi/180
    phie = phadf_rec[mask_adfrec][0]*np.pi/180+0.1*np.pi/180
    ct = np.cos(the)
    st = np.sin(the)
    cp = np.cos(phie)
    sp = np.sin(phie)
    ke = np.array([-st*cp, -st*sp, -ct])
    xce2 = func.compute_core(ke, xse2, cuts.refAlt)
    # Compute ellipse
    npts = 100
    v_cone = func.generate_cone_surface_vectors(ke, wcm, npts)
    x_elle2 = np.zeros((npts,3))
    for i, v in enumerate(v_cone):
        x_elle2[i,:] = func.compute_core(v, xse2, cuts.refAlt)



    fsuptit = (
    f"{df_info_rootfile['rootfile'][mask_info].iloc[0]} Event {merged['event_ID'][mask_coinc].iloc[0]}"
    )
    ftit = (
    rf"$\Theta = {thadf_rec[mask_adfrec][0]:.1f}^\circ, "
    rf"\Phi = {phadf_rec[mask_adfrec][0]:.1f}^\circ, "
    rf"\chi^2_\mathrm{{adf}} / \mathrm{{ndf}} = {reduced_chi2:.2f}$"
)

    ftit_pwf = (
    rf"$\chi^2_\mathrm{{pwf}} / \mathrm{{ndf}} = {reduced_chi2_pwf:.2f}$"
)
    
    ftit_background = (
    rf"$\Theta = {thadf_rec[mask_adfrec][0]:.1f}^\circ, "
    rf"\Phi = {phadf_rec[mask_adfrec][0]:.1f}^\circ, "
    rf"\chi^2_\mathrm{{background}} / \mathrm{{ndf}} = {chi2_red_backgroundfit:.2f}$"
)

    ftit_swf = rf"$\chi^2_{{\mathrm{{swf}}}}/\mathrm{{ndf}} = {reduced_chi2_swf:.2f}$" \
           rf"  $||X_{{\mathrm{{source}}}} - X_{{\mathrm{{core}}}}|| = {dist:.1f}\,\mathrm{{km}}$"
    


    # --- Plot ADF profiles for triggered DUs ---
    plt.figure()
    plt.plot(w_rec[mask_adf],am_rec[mask_adf],'ob',label="max ADC @ DU") 
    plt.errorbar(w_rec[mask_adf],am_rec[mask_adf],0.075*am_rec[mask_adf],fmt='None',marker='o',markerfacecolor='b')
    plt.plot(w_rec[mask_adf],af_rec[mask_adf],'+r',label="ADF fit @ DU")    
    plt.plot([wcm, wcm],[0,max(am_rec[mask_adf])*1.5],label="mean Cherenkov angle")
    
    # Compute mean ADF model for plotting
    w,adf_f = func.adf_fun(l,aadf_rec[mask_adfrec][0],wcm,dwadf_rec[mask_adfrec][0])
    plt.plot(w,adf_f,"--r",label="mean ADF model")
    #plt.ylim([0, max(am_rec[mask_adf])*1.5]) 
    plt.xlim([0,1.6])
    plt.xlabel("$\omega$ (deg)")
    plt.legend(loc="best")
    plt.suptitle(fsuptit, fontsize = 10)
    plt.title(ftit)
    plt.ylabel("Voltage (ADC)")
    #plt.savefig(f"{conf.output_figures}CR"+str(merged['id_coinc'][mask_coinc].iloc[0])+"_ADFprofile.png", bbox_inches='tight')
    #plt.close()
    plt.show()

    # --- Background fit plot ---
    sort_idx = np.argsort(l_ant_distance[mask_adf])

    x_sorted = l_ant_distance[mask_adf][sort_idx]
    y_sorted = A_backgroundfit[sort_idx]

    plt.figure()
    plt.scatter(l_ant_distance[mask_adf], am_rec[mask_adf], label='data')
    plt.plot(x_sorted, y_sorted, color='red', label='background model')
    plt.errorbar(l_ant_distance[mask_adf],am_rec[mask_adf],0.075*am_rec[mask_adf],fmt='None',marker='o',markerfacecolor='b')
    plt.suptitle(fsuptit, fontsize = 10)
    plt.title(ftit_background)
    plt.xlabel("Distance between Xsource and antenna [m]")
    plt.ylabel("Voltage [ADC]")
    plt.legend()
    #plt.savefig(f"{conf.output_figures}CR"+str(merged['id_coinc'][mask_coinc].iloc[0])+"_backgroundfit.png", bbox_inches='tight')
    #plt.close()
    plt.show()


    # --- Compute Plane Wave Fit (PWF) residuals ---
    params = np.deg2rad(theta_rec[mask_pwf][0]) , np.deg2rad(phi_rec[mask_pwf][0]) 
    t_pwf_model = func.PWF_model(params, xant_rec[mask_adf,:], cuts.refAlt, cuts.c_light)
    t_pwf_measured = merged['time'][mask_coinc] * 1e9 #ns
    t0_model = t_pwf_model[0]
    t0_meas = t_pwf_measured.iloc[0] 

    # Centered timings
    t_pwf_model_centered = t_pwf_model - t0_model
    t_pwf_measured_centered = t_pwf_measured - t0_meas

    res_pwf = t_pwf_measured - t_pwf_model
    t0_offset_pwf = np.mean(res_pwf)
    res_centered_pwf = res_pwf - t0_offset_pwf

    # --- Compute Spherical Wave Fit (SWF) residuals ---
    params = np.deg2rad(theta_rec_swf[mask_pwf][0]) , np.deg2rad(phi_rec_swf[mask_pwf][0]), r_xmax[mask_pwf][0], t_0[mask_pwf][0]
    t_swf_model = func.SWF_model(params, xant_rec[mask_adf,:], cuts.refAlt, cuts.c_light)
    t_swf_measured = merged['time'][mask_coinc] * 1e9 #ns
    t0_model_swf = t_swf_model[0]
    t0_meas_swf = t_swf_measured.iloc[0] 

    # Centered timings
    t_swf_model_centered = t_swf_model - t0_model_swf
    t_swf_measured_centered = t_swf_measured - t0_meas_swf

    #print('model', t_swf_model_centered)
    #print('measured', t_swf_measured_centered)

    res_swf = t_swf_measured - t_swf_model
    t0_offset_swf = np.mean(res_swf)
    res_centered_swf = res_swf - t0_offset_swf



    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # ---- Panel 1 : PWF ----
    axs[0].scatter(t_pwf_measured, res_centered_pwf, s=25, alpha=0.8)
    axs[0].errorbar(t_pwf_measured ,res_centered_pwf,yerr=5,fmt='None',marker='o',markerfacecolor='b')

    axs[0].axhline(0, ls='--', lw=1, color='gray')
    axs[0].grid(ls='--', alpha=0.3)
    axs[0].set_ylabel('Residuals PWF [ns]', fontsize=20)
    axs[0].set_title(ftit_pwf, fontsize=20, pad=8)

    # ---- Panel 2 : SWF ----
    axs[1].scatter(t_swf_measured, res_centered_swf, s=25, alpha=0.8)
    axs[1].errorbar(t_swf_measured ,res_centered_swf,yerr=5,fmt='None',marker='o',markerfacecolor='b')
    axs[1].axhline(0, ls='--', lw=1, color='gray')
    axs[1].grid(ls='--', alpha=0.3)
    axs[1].set_ylabel('Residuals SWF [ns]', fontsize=20)
    axs[1].set_xlabel('Measured time [ns]', fontsize=20)
    axs[1].set_title(ftit_swf, fontsize=20, pad=8)

    # ---- Global title ----
    fig.suptitle(fsuptit, fontsize=10, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    #plt.savefig(f"{conf.output_figures}CR"+str(merged['id_coinc'][mask_coinc].iloc[0])+"_delayplot.png", bbox_inches='tight')
    #plt.close()
    plt.show()
    
    # --- Load antenna positions ---
    file_path = conf.ANTENNA_FILE 
    column_names = ['antenna_ID', 'y', 'x', 'z']
    antenna_position = pd.read_csv(file_path, sep='\s+', names=column_names, header=None)
    antenna_position['antenna_ID'] = antenna_position['antenna_ID'].astype(int) 
    antenna_position = antenna_position[~antenna_position['antenna_ID'].between(0, 15)]

    # --- Select antennas triggered in this event ---
    du_ids = merged["antenna_ID"][mask_coinc]
    du_positions = antenna_position[antenna_position['antenna_ID'].isin(du_ids)]
    du_positions = du_positions.set_index('antenna_ID').loc[du_ids]
    #print(du_positions)

    # --- Build coordinate dictionary for all antennas ---
    coord_dict_full = antenna_position.set_index('antenna_ID')[['x','y','z']].to_dict('index')
    #print("IDs event full :", list(coord_dict_full.keys()))
    #print("IDs number :", len(coord_dict_full))

    # --- Extract triggered antenna IDs from CD file ---
    antenna_ids_event = []
    antenna_data_rows = df_cd[df_cd['fullID'] == idx]
    antenna_data = antenna_data_rows.iloc[0, 2:].dropna()
    #print(df_cd['fullID'])
    #print(antenna_data)

    for x in antenna_data:
        antenna_ids_event.extend(map(int, str(x).split()))
        print(antenna_ids_event)

    # --- Build dictionary for only triggered antennas in CD file---
    coord_dict = {ant_id: coord_dict_full[ant_id] 
                for ant_id in antenna_ids_event if ant_id in coord_dict_full}
    Xants_event = np.array([list(coord_dict[ant_id].values()) for ant_id in antenna_ids_event])

               
    # Plot triggered antenna position and peak amplitude
    # Also plot all the antennas that have not triggered (red cross)
    print("Dir PWF:",theta_rec[mask_pwf][0],phi_rec[mask_pwf][0])
    print("Dir ADF:",thadf_rec[mask_adfrec][0],phadf_rec[mask_adfrec][0])
    xs = xs_rec[mask_pwf,:][0]
    print("Point source:",xs)
    print("ADF fit parameters:", aadf_rec[mask_adfrec][0],dwadf_rec[mask_adfrec][0])

    ## Compute footprint
    mask_selected = antenna_position['antenna_ID'].isin(coord_dict.keys())
    x_sel = antenna_position[mask_selected]['x']
    y_sel = antenna_position[mask_selected]['y']
    ids_sel = antenna_position[mask_selected]['antenna_ID']

    x_other = antenna_position[~mask_selected]['x']
    y_other = antenna_position[~mask_selected]['y']

    plt.figure()
    sc = plt.scatter(-xant_rec[mask_adf,1], xant_rec[mask_adf,0], c=am_rec[mask_adf], cmap='viridis', s=am_rec[mask_adf], label="Triggered DUs") 
    #sc = plt.scatter(xant_rec[1], xant_rec[0], marker='+')    
    #antenna_position_filtered = antenna_position[~antenna_position['antenna_ID'].between(0, 15)]
    plt.scatter(antenna_position['y'], antenna_position['x'], marker ='+', color ='red') 
    plt.scatter(y_other, x_other, marker ='+', color ='red', label="Silent DUs")
    plt.scatter(y_sel, x_sel, s=70, marker ='+', color='blue', label="Alive DUs")
    #for xi, yi, aid in zip(-y_sel, x_sel, ids_sel):
    #    plt.text(xi, yi, str(aid), fontsize=8, ha='right', va='bottom')
    #plt.scatter(-y_all, x_all, cmap='viridis')   # Weird, does not match am_rec
    plt.colorbar(sc, label='Peak amplitude (ADC)')
       
    #plt.scatter(y_all, x_all,  marker='+', c='red')
    plt.arrow(-xc[1]+k[1]*1e3,xc[0]-k[0]*1e3, -k[1]*1e3, k[0]*1e3, head_width=1, head_length=1, fc='black', ec='black')
    #plt.arrow(xc[1]+k[1]*1e3,xc[0]-k[0]*1e3, -k[1]*1e3, k[0]*1e3, head_width=1, head_length=1, fc='black', ec='black')
    # Footprint
    plt.plot(-xc[1],xc[0],'ko')
    plt.plot(-x_ell[:,1],x_ell[:,0],'--k',linewidth=2)
    # Now footprint with error
    #plt.plot(-xce[1],xce[0],'ro',markerfacecolor='None') 
    #plt.plot(-x_elle[:,1],x_elle[:,0],'-.r')
    #plt.plot(-xce2[1],xce2[0],'ro',markerfacecolor='None') 
    #plt.plot(-x_elle2[:,1],x_elle2[:,0],'-.r')
    for i, du_id in enumerate(du_ids):
        plt.text(-xant_rec[mask_adf,1][i]+50, xant_rec[mask_adf,0][i]+50, f"DU{int(du_id)}", fontsize=8, color='white')

    #for i  in range(len(du_ids)):
        #plt.text(xant_rec[mask_adf,1][i]+50, xant_rec[mask_adf,0][i]+50, f"DU{merged['antenna_ID'][mask_coinc][i]}", fontsize=8)
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.grid(True)
    plt.legend()
    #plt.axis('equal')
    plt.ylim([xmin,xmax])
    plt.xlim([ymin,ymax])
    plt.suptitle(fsuptit, fontsize = 10)
    plt.title(ftit, x=0.6)
    plt.subplots_adjust(left=0.15) 
    #plt.close()
    plt.show()


    
