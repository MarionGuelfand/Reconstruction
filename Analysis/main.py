# main.py

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

# --------------------------
# Load antenna positions
# --------------------------
antenna_position = func.load_antenna_positions(
    conf.ANTENNA_FILE, 
    shower_core_height=cuts.ShowerCoreHeight)

# Create a dictionary with coordinates of each antenna (key: antenna_ID)
coord_dict_full = {
    row.antenna_ID: {'x': row.x, 'y': row.y, 'z': row.z}
    for _, row in antenna_position.iterrows()
}


# --------------------------
# Prepare dictionary to store results per month
# --------------------------
data_month = {}
for mn in conf.month_names:
    data_month[mn] = {
        'all_am_std': [],
        'all_am_mean': [],
        'all_am_max': [],
        'all_ratio_trig_total': [],
        'all_theta_adf': [],
        'all_theta_pwf': [],
        'all_phi_adf': [],
        'all_nants_adf': [],
        'all_reduced_chi2_adf': [],
        'all_theta_pwd': [],
        'all_nants_pwf': [],
        'all_reduced_chi2_swf': [],
        'all_reduced_chi2_pwf': [],
        'all_reduced_chi2_background': [],
        'all_w_adf_mean': [],
        'all_wc_adf_mean': [],
        'all_w_adf_std': [],
        'all_longitudinaldistance_Xsource': [],
        'all_grammage_Xsource': []
    }



# --------------------------
# Main loop over directories (months)
# --------------------------
for root_path, month_name in zip(conf.root_paths, conf.month_names):
    total_count = 0
    total_count_filter = 0 

    for subdir, dirs, files in os.walk(root_path):
        if 'Figures' in subdir:
            continue

        f_swf = os.path.join(subdir, 'Rec_sphere_wave_recons.txt')
        f_pwf = os.path.join(subdir, 'Rec_plane_wave_recons.txt')
        f_adf = os.path.join(subdir, 'Rec_adf_recons.txt')
        f_adff = os.path.join(subdir, 'Rec_adf_parameters.txt')
        f_info_rootfile = os.path.join(subdir, 'Info_rootfile.txt')
        f_antennas = os.path.join(subdir, 'Rec_coinctable.txt')
        f_coinc = os.path.join(subdir, 'coord_antennas_withID.txt')
        f_CD_all = os.path.join(subdir, 'events_CD_all_events.txt')

        # Skip if any required file is missing
        if not all(os.path.exists(f) for f in [f_swf, f_pwf, f_adf, f_adff, f_info_rootfile, f_antennas, f_coinc]):
            continue

        swf = np.atleast_2d(np.genfromtxt(f_swf, dtype=float, autostrip=True))
        pwf = np.atleast_2d(np.genfromtxt(f_pwf, dtype=float, autostrip=True))
        adf = np.atleast_2d(np.genfromtxt(f_adf, dtype=float, autostrip=True))
        adff = np.atleast_2d(np.genfromtxt(f_adff, dtype=float, autostrip=True))

        df_ant = pd.read_csv(f_antennas, sep=r"\s+", header=None,
                             names=["id", "id_coinc", "time", "amp_data", "event_ID", "run"])
        df_coord = pd.read_csv(f_coinc, sep=r"\s+", header=None,
                               names=["id", "x_antenna", "y_antenna", "z_antenna", "antenna_ID", "event_ID", "run"])
        merged = pd.merge(df_ant, df_coord, on=["id", "event_ID", "run"], suffixes=("_rec", "_coord"))

        df_cd = pd.read_csv(f_CD_all, sep="\t", header=None, engine='python')
        df_cd.rename(columns={0: "fullID", 1: "n_antennas"}, inplace=True)

        idpwf_rec = pwf[:,0]; nants_pwf_arr = pwf[:,1]; theta_pwf_arr = pwf[:,2]; chi2_pwf_arr = pwf[:,5]
        idswf_rec = swf[:,0]; chi2_swf_arr = swf[:,2]; nants_swf_arr = swf[:,1]; xs_rec = swf[:,4:7]; theta_swf_arr = swf[:,9]
        idadf_rec = adf[:,0]; nants_adf_arr = adf[:,1]; theta_adf_arr = adf[:,2]; phi_adf_arr = adf[:,4]; chi2_adf_arr = adf[:,6];  dwadf_rec = adf[:,8]; aadf_rec = adf[:,9]
        id_rec = adff[:,0]; c = adff[:,2]; w_rec = adff[:,5]; l_ant_distance = adff[:,8]; am_rec = adff[:,2]; wc_rec = adff[:,6]

        # --- Loop over unique coincident event IDs ---
        for idx in np.unique(merged['id_coinc']):
            mask_adf = (id_rec == idx)
            mask_adfrec = (idadf_rec == idx)
            mask_pwf = (idpwf_rec == idx)
            mask_swf = (idswf_rec == idx)

            l_sel = l_ant_distance[mask_adf]
            A_sel = am_rec[mask_adf]
            chi2_swf_sel = chi2_swf_arr[mask_swf]
            chi2_pwf_sel = chi2_pwf_arr[mask_pwf]
            chi2_adf_sel = chi2_adf_arr[mask_adfrec]

            xs_rec_sel = xs_rec[mask_swf]
            w_rec_sel = w_rec[mask_adf]
            wc_rec_sel = wc_rec[mask_adf]
            dwadf_sel = dwadf_rec[mask_adfrec]
            theta_adf_sel = theta_adf_arr[mask_adfrec]
            phi_adf_sel = phi_adf_arr[mask_adfrec]
            am_rec_sel = am_rec[mask_adf]

            reduced_chi2_adf = func.compute_reduced_chi2(chi2_adf_arr[mask_adfrec], nants_adf_arr[mask_adfrec], n_params=4)[0]
            reduced_chi2_swf = func.compute_reduced_chi2(chi2_swf_arr[mask_pwf], nants_swf_arr[mask_pwf], n_params=4)[0]
            reduced_chi2_pwf = func.compute_reduced_chi2(chi2_pwf_arr[mask_pwf], nants_pwf_arr[mask_pwf], n_params=2)[0]

            total_count += 1 

            # --- Apply cuts ---
            if (np.isnan(l_sel).any() or np.isnan(A_sel).any() or
                np.isnan(chi2_swf_sel).any() or np.isnan(chi2_pwf_sel).any() or np.isnan(chi2_adf_sel).any() or
                np.isinf(l_sel).any() or np.isinf(A_sel).any() or
                np.isinf(chi2_swf_sel).any() or np.isinf(chi2_pwf_sel).any() or (l_sel == 0).any()):
                continue
            if (w_rec_sel > cuts.cuts["w_rec_max"]).any():
                continue

            if (theta_adf_sel >= cuts.cuts["theta_max"]).any():
                continue

            if reduced_chi2_adf > cuts.cuts["reduced_chi2_adf_max"]:
                continue

            if reduced_chi2_swf > cuts.cuts["reduced_chi2_swf_max"]:
                continue

            # --- Omega / background calculations ---
            A_backgroundfit, chi2_red_backgroundfit = func.background_fit(A_sel, l_sel)

            # --- Compute omega angles for triggered antennas and compute trigger ratio ---
            theta = theta_adf_sel[0]
            phi = phi_adf_sel[0]
            Xmax = xs_rec_sel[0]
            Xants = antenna_position[['x','y','z']].values
            omega_all_antenna = func.compute_omega(theta, phi, Xants, Xmax)
            wc_mean = np.mean(wc_rec_sel)
            bin_width = cuts.cuts["bin_width"]

            antenna_ids_event = []
            antenna_data_rows = df_cd[df_cd['fullID'] == idx]
            antenna_data = antenna_data_rows.iloc[0, 2:].dropna()
            for x in antenna_data:
                antenna_ids_event.extend(map(int, str(x).split()))
            
            coord_dict = {ant_id: coord_dict_full[ant_id] 
                          for ant_id in antenna_ids_event if ant_id in coord_dict_full}
            Xants_event = np.array([list(coord_dict[ant_id].values()) for ant_id in antenna_ids_event])
            omega_event = func.compute_omega(theta, phi, Xants_event, Xmax)

            mask_total = (omega_event >= wc_mean - bin_width) & (omega_event <= wc_mean + bin_width)
            mask_trig  = (w_rec_sel >= wc_mean - bin_width) & (w_rec_sel <= wc_mean + bin_width)
            N_total = np.sum(mask_total)
            N_trig  = np.sum(mask_trig)
            ratio_trig_total = N_trig / N_total if N_total > 0 else 0

            # --- Reconstruct grammage / distance ---
            n_events = np.atleast_1d(theta_adf_sel).size
            InjectionHeight = np.full(n_events, cuts.InjectionHeight)
            ShowerCoreHeight = np.full(n_events, cuts.ShowerCoreHeight)
            AzimuthRec_GRAND = phi_adf_sel
            ZenithRec_GRAND = theta_adf_sel
            k = func.compute_shower_axis(theta_adf_sel, phi_adf_sel)
            xs = xs_rec_sel
            xc = func.compute_core(k, xs, cuts.ShowerCoreHeight)
            dist = np.linalg.norm(xs - xc) / 1000 # km
            Grammage_recons, LongitudinalDistance_Source = func.recons_grammage(
                AzimuthRec_GRAND, ZenithRec_GRAND, xs_rec_sel[:,0], xs_rec_sel[:,1], xs_rec_sel[:,2],
                InjectionHeight, ShowerCoreHeight, xc[0], xc[1]
            )

            total_count_filter += 1

            # --- Store results for this month ---
            data_month[month_name]['all_am_std'].append(np.std(am_rec_sel))
            data_month[month_name]['all_am_mean'].append(np.mean(am_rec_sel))
            data_month[month_name]['all_am_max'].append(np.max(am_rec_sel))
            data_month[month_name]['all_ratio_trig_total'].append(ratio_trig_total)
            data_month[month_name]['all_theta_adf'].append(theta_adf_sel[0])
            data_month[month_name]['all_phi_adf'].append(phi_adf_sel[0])
            data_month[month_name]['all_theta_pwf'].append(theta_pwf_arr[mask_pwf][0])
            data_month[month_name]['all_nants_adf'].append(nants_adf_arr[mask_adfrec][0])
            data_month[month_name]['all_nants_pwf'].append(nants_pwf_arr[mask_pwf][0])
            data_month[month_name]['all_reduced_chi2_adf'].append(reduced_chi2_adf)
            data_month[month_name]['all_reduced_chi2_swf'].append(reduced_chi2_swf)
            data_month[month_name]['all_reduced_chi2_pwf'].append(reduced_chi2_pwf)
            data_month[month_name]['all_reduced_chi2_background'].append(chi2_red_backgroundfit)
            data_month[month_name]['all_w_adf_mean'].append(np.mean(w_rec[mask_adf]))
            data_month[month_name]['all_w_adf_std'].append(np.std(w_rec[mask_adf]))
            data_month[month_name]['all_wc_adf_mean'].append(np.mean(wc_rec[mask_adf]))
            data_month[month_name]['all_grammage_Xsource'].append(Grammage_recons)
            data_month[month_name]['all_longitudinaldistance_Xsource'].append(dist)

    # --- Print statistics per month ---
    #print(f"{month_name}: {total_count} events total")
    print(f"{month_name}: {total_count_filter} events passed all cuts")
    #print(f"{month_name}: {total_count_filter} out of {total_count} events passed all cuts "
    #  f"({100 * total_count_filter/total_count:.2f}%)")


all_theta_adf_all = np.concatenate([data_month[m]['all_theta_adf'] for m in data_month])
all_phi_adf_all = np.concatenate([data_month[m]['all_phi_adf'] for m in conf.month_names])
all_theta_pwf_all = np.concatenate([data_month[m]['all_theta_pwf'] for m in data_month])
all_nants_adf_all = np.concatenate([data_month[m]['all_nants_adf'] for m in data_month])
all_nants_pwf_all = np.concatenate([data_month[m]['all_nants_pwf'] for m in data_month])
all_reduced_chi2_adf_all = np.concatenate([data_month[m]['all_reduced_chi2_adf'] for m in data_month])
all_reduced_chi2_swf_all = np.concatenate([data_month[m]['all_reduced_chi2_swf'] for m in data_month])
all_reduced_chi2_pwf_all = np.concatenate([data_month[m]['all_reduced_chi2_pwf'] for m in data_month])
all_grammage_Xsource = np.concatenate([data_month[m]['all_grammage_Xsource'] for m in data_month])
all_longitudinaldistance_Xsource = np.concatenate([data_month[m]['all_longitudinaldistance_Xsource'] for m in data_month])

# --- Log-spaced bins for SWF, PWF, and ADF histograms ---
bins_log_swf = np.logspace(np.log10(all_reduced_chi2_swf_all[all_reduced_chi2_swf_all>0].min()), 
                           np.log10(all_reduced_chi2_swf_all[all_reduced_chi2_swf_all>0].max()), 20)
bins_log_pwf = np.logspace(np.log10(all_reduced_chi2_pwf_all[all_reduced_chi2_pwf_all>0].min()), 
                           np.log10(all_reduced_chi2_pwf_all[all_reduced_chi2_pwf_all>0].max()), 20)
bins_log_adf = np.logspace(np.log10(all_reduced_chi2_adf_all[all_reduced_chi2_adf_all>0].min()), 
                           np.log10(all_reduced_chi2_adf_all[all_reduced_chi2_adf_all>0].max()), 20)



# Scatter plot: longitudinal distance vs grammage
plt.figure()
plt.scatter(all_longitudinaldistance_Xsource, all_grammage_Xsource, alpha=0.7)
plt.xlabel(rf"  $|X_{{\mathrm{{source}}}} - X_{{\mathrm{{core}}}}| [\,\mathrm{{km}}]$")
plt.ylabel(r'$X_{\mathrm{source}}\ \mathrm{[g/cm^2]}$')
#plt.savefig(f'{output_figures}Xmax_grammage.pdf')
plt.show()

# Histogram: longitudinal distance
mean_long = np.mean(np.array(all_longitudinaldistance_Xsource))
std_long  = np.std(np.array(all_longitudinaldistance_Xsource))
plt.figure()
plt.hist(all_longitudinaldistance_Xsource, alpha=1, bins=20)
plt.xlabel(rf"  $|X_{{\mathrm{{source}}}} - X_{{\mathrm{{core}}}}| [\,\mathrm{{km}}]$")
#plt.savefig(f'{output_figures}Xmax_dist_histo.pdf')
plt.show()

# Histogram: grammage
mean_grammage = np.mean(np.array(all_grammage_Xsource))
std_grammage  = np.std(np.array(all_grammage_Xsource))
plt.figure()
plt.hist(all_grammage_Xsource, alpha=1, bins=20)
plt.xlabel(r'$X_{\mathrm{source}}\ \mathrm{[g/cm^2]}$')
#plt.savefig(f'{output_figures}Xmax_grammage_histo.pdf')
plt.show()

# Scatter plot: theta (ADF) vs longitudinal distance
plt.figure()
plt.scatter(all_theta_adf_all, all_longitudinaldistance_Xsource, alpha=0.7)
plt.xlabel(r"$\theta_\mathrm{ADF}$ [°]")
plt.ylabel(rf"  $|X_{{\mathrm{{source}}}} - X_{{\mathrm{{core}}}}| [\,\mathrm{{km}}]$")
plt.yscale('log')
plt.ylim(1e0, 5e2)
#plt.savefig(f'{output_figures}Xmax_dist_theta.pdf')
plt.show()

# Scatter plot: SWF chi² vs grammage
plt.figure()
plt.scatter(all_reduced_chi2_swf_all, all_grammage_Xsource, alpha=0.7)
plt.xlabel(r"$\chi^2_{\mathrm{swf}}/\mathrm{ndf}$")
plt.ylabel(r'$X_{\mathrm{source}}\ \mathrm{[g/cm^2]}$')
#plt.savefig(f'{output_figures}Xmax_dist_chi2swf.pdf')
plt.show()

# Scatter plot: number of antennas vs grammage
plt.figure()
plt.scatter(all_nants_adf_all, all_grammage_Xsource, alpha=0.7)
plt.xlabel('Nants')
plt.ylabel(r'$X_{\mathrm{source}}\ \mathrm{[g/cm^2]}$')
plt.show()

# Scatter plot: SWF chi² vs theta (ADF)
plt.figure()
plt.scatter(all_reduced_chi2_swf_all, all_theta_adf_all, alpha=0.7)
plt.xlabel(r"$\chi^2_{\mathrm{swf}}/\mathrm{ndf}$")
plt.ylabel(r"$\theta_\mathrm{ADF}$ [°]")
plt.show()

# Scatter plot: theta (ADF) vs grammage
plt.figure()
plt.scatter(all_theta_adf_all, all_grammage_Xsource, alpha=0.7)
plt.xlabel(r"$\theta_\mathrm{ADF}$ [°]")
plt.ylabel(r'$X_{\mathrm{source}}\ \mathrm{[g/cm^2]}$')
#plt.savefig(f'{output_figures}Gramma_Xmax_theta.pdf')
plt.show()


# Histogram: PWF chi² (all months combined)
vals_pwf = all_reduced_chi2_pwf_all[all_reduced_chi2_pwf_all > 0]
plt.figure()
plt.hist(vals_pwf, bins=bins_log_pwf, alpha=0.7)
plt.xscale('log')
plt.xlabel(r"$\chi^2_{\mathrm{pwf}}/\mathrm{ndf}$")
plt.ylabel('Counts')
#plt.xlim(1e-4, 1e5)
#plt.title("Global Histogram – χ² pwf (all months combined)")
plt.show()

# Histogram: PWF chi² per month
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    vals = np.array(data_month[month_name]['all_reduced_chi2_pwf'])
    vals = vals[vals > 0]
    plt.hist(vals, bins=bins_log_pwf, alpha=0.5, color=color, label=month_name)
plt.xscale('log')
plt.xlabel(r"$\chi^2_{\mathrm{pwf}}/\mathrm{ndf}$")
plt.ylabel('Counts')
plt.grid(True, which='both', alpha=0.3)
#plt.xlim(1e-2, 1e5)
plt.legend()
plt.show()


# Histogram: SWF chi² (all months combined)
vals_swf = all_reduced_chi2_swf_all[all_reduced_chi2_swf_all > 0]
plt.figure()
plt.hist(vals_swf, bins=bins_log_swf, alpha=1)
plt.axvline(100, color='black', linestyle='--', linewidth=1.5)
plt.xscale('log')
plt.xlabel(r"$\chi^2_{\mathrm{swf}}/\mathrm{ndf}$")
plt.ylabel('Counts')
#plt.xlim(1e-4, 1e5)
#plt.title("Global Histogram – χ² swf (all months combined)")
#plt.savefig(f'{output_figures}SWF_all_theta_omega_cut.pdf')
plt.show()

# Histogram: SWF chi² per month
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    vals = np.array(data_month[month_name]['all_reduced_chi2_swf'])
    vals = vals[vals > 0]
    plt.hist(vals, bins=bins_log_swf, alpha=0.5, color=color, label=month_name)
plt.xscale('log')
plt.xlabel(r"$\chi^2_{\mathrm{swf}}/\mathrm{ndf}$")
plt.ylabel('Counts')
#plt.xlim(1e-2, 1e5)
plt.legend()
plt.show()

# Histogram: ADF chi² (all months combined)
vals_adf = all_reduced_chi2_adf_all[all_reduced_chi2_adf_all > 0]
plt.figure()
plt.hist(vals_adf, bins=bins_log_adf, alpha=1)
plt.axvline(20, color='black', linestyle='--', linewidth=1.5)
plt.xscale('log')
plt.xlabel(r"$\chi^2_{\mathrm{adf}}/\mathrm{ndf}$")
plt.ylabel('Counts')
#plt.savefig(f'{output_figures}ADF_all_theta_omega_cut.pdf')
#plt.xlim(1e-2, 5e3)
plt.show()

# Histogram: ADF chi² per month
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    vals = np.array(data_month[month_name]['all_reduced_chi2_adf'])
    vals = vals[vals > 0]
    plt.hist(vals, bins=bins_log_adf, alpha=0.5, color=color, label=month_name)
plt.xscale('log')
plt.xlabel(r"$\chi^2_{\mathrm{adf}}/\mathrm{ndf}$")
plt.ylabel('Counts')
plt.grid(True, which='both', alpha=0.3)
#plt.xlim(1e-2, 5e3)
plt.legend()
plt.show()

# Scatter plot: ADF chi² vs background chi² (log-log) per month
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    plt.scatter(data_month[month_name]['all_reduced_chi2_adf'],
                np.array(data_month[month_name]['all_reduced_chi2_background']),
                alpha=0.5, label=month_name, color=color, s=10)
    
xmin, xmax = 1e-1, 50
ymin, ymax = 1e-1, 1e3
x_line = np.linspace(xmin, xmax, 1000)
plt.plot(x_line, x_line, 'k--', linewidth=1, label="x = y")
plt.xlabel(r"$\chi^2_{\mathrm{adf}}/\mathrm{ndf}$")
plt.ylabel(r"$\chi^2_{\mathrm{background}}/\mathrm{ndf}$")
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()

# Histogram: Trigger ratio (%) per month
all_vals = np.concatenate([np.array(data_month[m]['all_ratio_trig_total'])*100 
                           for m in conf.month_names])
all_vals = all_vals[all_vals > 0]
bins_lin = np.linspace(0, 100, 20)

plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    vals = np.array(data_month[month_name]['all_ratio_trig_total'])*100
    vals = vals[vals > 0]
    plt.hist(vals, alpha=0.5, bins=bins_lin, color=color, label=month_name)
plt.xlabel(rf"Ratio $N_\mathrm{{triggered}} / N_\mathrm{{alive}}$ around $\omega_c \pm {bin_width} ^\circ$")
plt.ylabel('Counts')
plt.xlim(all_vals.min(), 100)
plt.legend()
#plt.savefig(f'{output_figures}Histo_Nants_after_cut.pdf')
plt.show()

# Scatter plot: Std amplitude vs trigger ratio
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    plt.scatter(data_month[month_name]['all_am_std'],
                np.array(data_month[month_name]['all_ratio_trig_total'])*100,
                alpha=0.5, label=month_name, color=color, s=10)
plt.xlabel("Std Measured Amplitude")
plt.ylabel(rf"Ratio $N_\mathrm{{triggées}} / N_\mathrm{{total}}$ around $\omega_c \pm {bin_width} ^\circ$")
plt.grid(True, which='both', alpha=0.3)
plt.ylim(-5, 105)
plt.legend()
plt.show()

# Scatter plot: Mean amplitude vs trigger ratio
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    plt.scatter(data_month[month_name]['all_am_mean'],
            np.array(data_month[month_name]['all_ratio_trig_total'])*100,
            alpha=0.5, label=month_name, color=color, s=10)
plt.xlabel("Mean peak Amplitude on triggered DUs [ADC]")
plt.ylabel(rf"Ratio $N_\mathrm{{triggered}} / N_\mathrm{{total}}$ around $\omega_c \pm {bin_width} ^\circ$")
plt.grid(True, which='both', alpha=0.3)
plt.ylim(-5, 105)
plt.legend()
plt.show()

# Scatter plot: Max amplitude vs trigger ratio
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    plt.scatter(data_month[month_name]['all_am_max'],
            np.array(data_month[month_name]['all_ratio_trig_total'])*100,
            alpha=0.5, label=month_name, color=color, s=10)
plt.xlabel("Max peak amplitude on triggered DUs [ADC]")
plt.ylabel(rf"Ratio $N_\mathrm{{triggered}} / N_\mathrm{{total}}$ around $\omega_c \pm {bin_width} ^\circ$")
plt.grid(True, which='both', alpha=0.3)
plt.ylim(-5, 105)
plt.legend()
plt.show()

bins_theta = np.linspace(all_theta_adf_all.min(), all_theta_adf_all.max(), 20)
bins_phi = np.linspace(all_phi_adf_all.min(), all_phi_adf_all.max(), 20)

# Histogram: theta (ADF) stacked, all months
plt.figure()
plt.hist(all_theta_adf_all, bins=bins_theta, stacked=True, alpha=1)
plt.xlabel(r"$\theta_\mathrm{ADF}$ [°]")
#plt.savefig(f'{output_figures}Histo_theta_after_cut.pdf')
plt.show()


# Histogram: theta (ADF) per month
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    plt.hist(data_month[month_name]['all_theta_adf'],
                alpha=0.3, label=month_name, color=color, bins=bins_theta)
plt.xlabel(r"$\theta_\mathrm{ADF}$ [°]")
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.show()

# Histogram: phi (ADF) stacked, all months
plt.figure()
plt.hist(all_phi_adf_all, bins=bins_phi, stacked=True, alpha=1)
plt.xlabel(r"$\phi_\mathrm{ADF}$ [°]")
#plt.savefig(f'{output_figures}Histo_phi_after_cut.pdf')
plt.show()


# Histogram: phi (ADF) per month
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    plt.hist(data_month[month_name]['all_phi_adf'],
                alpha=0.3, label=month_name, color=color, bins=bins_phi)
plt.xlabel(r"$\phi_\mathrm{ADF}$ [°]")
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.show()

# Histogram: number of antennas stacked
plt.figure()
bins_nants = np.arange(all_nants_adf_all.min(), all_nants_adf_all.max() + 2) - 0.5  # centré sur l'entier
plt.hist(all_nants_adf_all, bins=bins_nants, stacked=True, alpha=1)
plt.xlabel("Number of antennas (N$_\mathrm{ants}$)")
plt.xticks(np.arange(all_nants_adf_all.min(), all_nants_adf_all.max() + 1, 1))
#plt.yscale('log')
#plt.savefig(f'{output_figures}Histo_Nants_after_cut.pdf')
plt.show()

# Histogram: number of antennas per month
all_nants = np.concatenate([data_month[m]['all_nants_adf'] for m in conf.month_names])
bins_nants = np.linspace(all_nants.min(), all_nants.max(), 10)

plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    plt.hist(data_month[month_name]['all_nants_adf'],
                alpha=0.3, label=month_name, color=color, bins=bins_nants)
plt.xlabel("Number of antennas (N$_\mathrm{ants}$)")
plt.grid(True, which='both', alpha=0.3)
#plt.xlim(0,25)
plt.yscale('log')
plt.legend()
plt.show()


# --- Theta ADF vs Ratio ---
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    plt.scatter(data_month[month_name]['all_theta_adf'],
                np.array(data_month[month_name]['all_ratio_trig_total'])*100,
                alpha=0.3, label=month_name, color=color, s=10)
plt.xlabel(r"$\theta_\mathrm{ADF}$ [°]")
plt.ylabel(rf"Ratio $N_\mathrm{{triggered}} / N_\mathrm{{total}}$ around $\omega_c \pm {bin_width} ^\circ$")
plt.grid(True, which='both', alpha=0.3)
plt.ylim(-5, 105)
plt.legend()
plt.show()

# --- Nants vs Ratio ---
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    plt.scatter(data_month[month_name]['all_nants_adf'],
                np.array(data_month[month_name]['all_ratio_trig_total'])*100,
                alpha=0.3, label=month_name, color=color, s=10)
plt.xlabel("Number of antennas (N$_\mathrm{ants}$)")
plt.ylabel(rf"Ratio $N_\mathrm{{triggered}} / N_\mathrm{{total}}$ around $\omega_c \pm {bin_width} ^\circ$")
plt.grid(True, which='both', alpha=0.3)
plt.ylim(-5, 105)
plt.legend()
plt.show()

# --- N_ant vs Reduced Chi2 ADF ---
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    plt.scatter(data_month[month_name]['all_nants_adf'],
                data_month[month_name]['all_reduced_chi2_adf'],
                alpha=0.5, label=month_name, color=color, s=10)
plt.xlabel("Number of antennas (N$_\mathrm{ants}$)")
plt.ylabel(r"$\chi^2_\mathrm{adf}/\mathrm{ndf}$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# --- N_ant vs  Theta ---
plt.figure()
for month_name, color in zip(conf.month_names, conf.colors):
    plt.scatter(data_month[month_name]['all_nants_adf'],
                data_month[month_name]['all_theta_adf'],
                alpha=0.3, label=month_name, s=10, color=color)
plt.xlabel("Number of antennas (N$_\mathrm{ants}$)")
plt.ylabel(r"$\theta_\mathrm{ADF}$ [°]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# Concatenate theta and phi values from all months for a global ADF skyplot
theta_all = np.concatenate([data_month[m]['all_theta_adf'] for m in conf.month_names])
phi_all   = np.concatenate([data_month[m]['all_phi_adf'] for m in conf.month_names])  # Assure-toi que 'all_phi_adf' existe

# Convert angles to radians for matplotlib polar plot
# Phi: azimuth (0° = North, counterclockwise)
azimuth_rad = np.deg2rad(phi_all)

# Theta: zenith angle (0° = zenith at center)
zenith_rad = np.deg2rad(theta_all)

# --- Plot ---
plt.figure(figsize=(7,7))
ax = plt.subplot(projection='polar')

sc = ax.scatter(azimuth_rad, zenith_rad, s=30, alpha=0.8, label='ADF reconstruction')

ax.set_theta_zero_location('N')  # North at top
ax.set_theta_direction(1)        # counterclockwise

ax.set_ylim(0, np.pi/2)          # 0° zenith at center, 90° at edge

# Azimuth ticks (0°–360° every 45°)
ticks_deg = np.arange(0, 360, 45)
ticks_rad = np.deg2rad(ticks_deg)
ax.set_xticks(ticks_rad)
ax.set_xticklabels([f"{int(t)}°" for t in ticks_deg])

# Zenith ticks (0°–90° every 15°)
r_ticks_deg = np.arange(0, 91, 15)
r_ticks_rad = np.deg2rad(r_ticks_deg)
ax.set_yticks(r_ticks_rad)
ax.set_yticklabels([f"{int(t)}°" for t in r_ticks_deg])

for label in ax.get_xticklabels():
    label.set_horizontalalignment('center')
    label.set_verticalalignment('top')
    label.set_fontsize(14)
    label.set_y(label.get_position()[1] - 0.03)

for label in ax.get_yticklabels():
    label.set_horizontalalignment('right')
    label.set_fontsize(14)
    label.set_x(-0.1)

ax.set_rlabel_position(135)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)

plt.tight_layout()
plt.show()

# --- Polar plot: data separated by month ---
plt.figure(figsize=(7,7))
ax = plt.subplot(projection='polar')

for month_name, color in zip(conf.month_names, conf.colors):
    theta_all = np.array(data_month[month_name]['all_theta_adf'])
    phi_all   = np.array(data_month[month_name]['all_phi_adf']) 

    azimuth_rad = np.deg2rad(phi_all)   
    zenith_rad  = np.deg2rad(theta_all)      

    ax.scatter(azimuth_rad, zenith_rad, s=10, alpha=0.8, color=color, label=month_name)

# Same polar plot settings as above
ax.set_theta_zero_location('N')  
ax.set_theta_direction(1)         
ax.set_ylim(0, np.pi/2)

ticks_deg = np.arange(0, 360, 45)
ticks_rad = np.deg2rad(ticks_deg)
ax.set_xticks(ticks_rad)
ax.set_xticklabels([f"{int(t)}°" for t in ticks_deg])

r_ticks_deg = np.arange(0, 91, 15)
r_ticks_rad = np.deg2rad(r_ticks_deg)
ax.set_yticks(r_ticks_rad)
ax.set_yticklabels([f"{int(t)}°" for t in r_ticks_deg])

for label in ax.get_xticklabels():
    label.set_horizontalalignment('center')
    label.set_verticalalignment('top')
    label.set_fontsize(14)
    label.set_y(label.get_position()[1] - 0.03)

for label in ax.get_yticklabels():
    label.set_horizontalalignment('right')
    label.set_fontsize(14)
    label.set_x(-0.1)

ax.set_rlabel_position(135)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)

plt.tight_layout()
#plt.savefig(f'{output_figures}skyplot_aftercuts.pdf')
plt.show()


