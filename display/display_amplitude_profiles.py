#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot amplitude profiles for cosmic ray events reconstructed with ADF.


This script reads ADF reconstruction outputs, selects events with a reduced chi-squared
value ≤ 25 (considered as cosmic ray candidates), and generates amplitude profiles
comparing simulated and reconstructed values for each selected event.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # --- Paths ---
    #output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_DC2/CR_candidates/efield/new/noZchannel/'
    output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_DC2/CR_candidates/voltage_chi2adf_incertitude/'
    figures_dir = os.path.join(output_directory, 'Figures')
    os.makedirs(figures_dir, exist_ok=True)

    # --- Read input data ---
    merged_df = pd.read_csv(
        f'{output_directory}Rec_adf_parameters.txt',
        sep=r'\s+',
        names=[
            "EventName", "Nants", "AmpData", "AmpModel", "eta", "omega", "omega_c",
            "omega_c_analytic", "l", "x_antenna", "y_antenna", "z_antenna"
        ]
    )

    tab_adf = pd.read_csv(
        f'{output_directory}Rec_adf_recons.txt',
        sep=r'\s+',
        names=[
            "EventName", "nants", "ZenithRec", "nan", "AzimuthRec", "nanan",
            "Chi2", "NaN", "WidthRec", "AmpRec"
        ]
    )

    # --- Compute reduced chi-squared ---
    tab_adf['Chi2_reduced'] = tab_adf['Chi2'] / (tab_adf['nants'] - 4)

    # --- Filter events ---
    filtered_events_df = tab_adf[tab_adf['Chi2_reduced'] <= 25]

    # --- Plot for each filtered event ---
    for _, row in filtered_events_df.iterrows():
        event = row['EventName']
        chi2_red = row['Chi2_reduced']

        df_event = merged_df[merged_df['EventName'] == event]
        if df_event.empty:
            continue

        plt.figure(figsize=(6, 4))
        plt.scatter(df_event['omega'], df_event['AmpData'],
                    c='blue', label=r"Data", s=50)
        plt.scatter(df_event['omega'], df_event['AmpModel'],
                    marker='+', c='red', label=r"ADF model", s=50)
        plt.xlabel(r'$\omega$ [°]')
        plt.ylabel(r'Amplitude [ADC]')
        plt.title(f'Event {int(event)} — $\\chi^2_{{\\rm red}}$ = {chi2_red:.2f}')
        plt.legend(loc='lower center')
        plt.tight_layout()
        plt.show()

        # --- Save figure ---
        #plt.savefig(os.path.join(figures_dir, f'amplitude_profile_{int(event)}.png'), dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
