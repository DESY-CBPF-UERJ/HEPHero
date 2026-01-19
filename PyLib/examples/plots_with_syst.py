# %%
#==================================================================================================
# Import packages
#==================================================================================================
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import hepherolib.data as data
import hepherolib.analysis as ana
ana.start()


# %%
#==================================================================================================
# Read files
#==================================================================================================
basedir = "/home/gcorreia/cernbox_temp/HEP/OUTPUT/HHDM/Test/datasets"
period = '17'
luminosity = '41.5'
smooth_factor = 0.2

ds = data.read_files(basedir, period, mode="syst")


data.join_datasets(ds, "ST", [
    "ST_tW_antitop", 
    "ST_tW_top", 
    #"ST_s-channel",
    "ST_t-channel_top", 
    "ST_t-channel_antitop",
    ], mode="syst")

data.join_datasets(ds, "TT", [
    "TTTo2L2Nu", 
    "TTToSemiLeptonic",
    ], mode="syst")

data.join_datasets(ds, "ZZ", [
    #"ZZ_Inclusive"
    "ZZTo4L", 
    "ZZTo2L2Nu",
    ], mode="syst")

data.join_datasets(ds, "WZ", [
    #"WZ_Inclusive"
    "WZTo3LNu", 
    ], mode="syst")


data.join_datasets(ds, "DYJetsToLL", [
    "DYJetsToLL_Pt-Inclusive",
    #"DYJetsToLL_Pt-0To65",
    #"DYJetsToLL_Pt-50To100",
    #"DYJetsToLL_Pt-100To250",
    #"DYJetsToLL_Pt-250To400",
    #"DYJetsToLL_Pt-400To650",
    #"DYJetsToLL_Pt-650ToInf",
    ], mode="syst")

data.join_datasets(ds, "Residual", [
    "WZZ", 
    "WWZ", 
    "ZZZ", 
    "WWW", 
    "WW",
    "WZ_Others",
    "ZZ_Others",
    ], mode="syst")

data.join_datasets(ds, "Data", [ # It changes for each year
    "Data_B", 
    "Data_C", 
    "Data_D", 
    "Data_E", 
    "Data_F",
    ], mode="syst")

with open(os.path.join(basedir, "lateral_systematics.json")) as json_sys_file:
    systematics = json.load(json_sys_file)
with open(os.path.join(basedir, "vertical_systematics.json")) as json_sys_file:
    systematics.update(json.load(json_sys_file))
print(systematics)
#del systematics["ISR"]

def signal_label(param_0, param_1):
    label = r'$m_H=$' + str(param_0) + r', $m_a=$' + str(param_1)
    return label

#-------------------------------------------------------------------------------------------

bkg_colors = ['gainsboro', 'orchid', 'limegreen', 'darkgoldenrod', 'red', 'skyblue']
bkg_labels = ['Residual SM', r'$WZ$', r'$ZZ$', 'Drell-Yan', 'Single top', r'$t\bar{t}$']
backgrounds = [ds["Residual"], ds["WZ"], ds["ZZ"], ds["DYJetsToLL"], ds["ST"], ds["TT"]]

sgn_color = "blue"
sgn_label = [signal_label(1000, 800), signal_label(800, 600), signal_label(600, 400), signal_label(500, 300), signal_label(400, 200)]
signal = ["Signal_1000_800", "Signal_800_600", "Signal_600_400", "Signal_500_300", "Signal_400_200"]

outdir = os.path.join(basedir, period)
tag = "DYCR_not_boosted"
mode = "counting"
region = 1
var = "MLP_score_400_200"
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]

for i in range(len(signal)):

    harvest = ana.harvester(region, var, bins, backgrounds, bkg_labels, bkg_colors, ds[signal[i]], sgn_label[i], sgn_color, systematics, smooth_factor=smooth_factor, analysis_name="bbZa")
    harvest.set_data(ds["Data"])
    #datacard = harvest.get_combine_datacard(outdir, tag, mode=mode)

    # %%
    #==================================================================================================
    # Plot
    #==================================================================================================
    fig1 = plt.figure(figsize=(20,9))
    grid = [2, 3]
    gs1 = gs.GridSpec(grid[0], grid[1], height_ratios=[4, 1])  

    #=================================================================================================================
    N = 1
    #=================================================================================================================
    #==================================================
    ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
    #==================================================
    hist_cv, unc_up, unc_down = harvest.stacked_plot(ax1, width="same")
    hist_data, unc_data = harvest.data_plot(ax1, width="same")
    harvest.signal_plot(ax1, width="same")
    ana.labels(ax1, ylabel="Events")  # Set up the label names
    ana.style(ax1, lumi=luminosity, year=period[-2:], ylog=True, legend_ncol=4, ylim=[1.e-1,1.e7]) # Set up the plot style and information on top

    #==================================================
    ax2 = plt.subplot(ana.position(gs1,grid,N,2), sharex=ax1)  
    #==================================================
    harvest.ratio_plot(ax2, width="same")
    ana.labels(ax2, xlabel="Bin index of "+var, ylabel="Data / Bkg.")  
    ana.style(ax2, ylim=[0., 2], yticks=[0, 0.5, 1, 1.5, 2], xgrid=True, ygrid=True)

    #=================================================================================================================
    N = 2
    #=================================================================================================================
    #==================================================
    ax1 = plt.subplot(ana.position(gs1,grid,N,1))
    #==================================================
    harvest.frac_bkg_syst_plot(ax1, version=1, width="same")
    ana.labels(ax1, xlabel="Bin index of "+var, ylabel="Background Fractional Uncertainty")
    ana.style(ax1, ylog=True, ylim=[1.e-3, 4.e0], xgrid=True, ygrid=True, legend_ncol=5)
    
    #=================================================================================================================
    N = 3
    #=================================================================================================================
    #==================================================
    ax1 = plt.subplot(ana.position(gs1,grid,N,1))
    #==================================================
    harvest.frac_signal_syst_plot(ax1, version=1, width="same")
    ana.labels(ax1, xlabel="Bin index of "+var, ylabel="Signal Fractional Uncertainty")
    ana.style(ax1, ylog=True, ylim=[1.e-3, 4.e0], xgrid=True, ygrid=True, legend_ncol=5)

    #=================================================================================================================
    # Make final setup, save and show plots
    #=================================================================================================================
    plt.subplots_adjust(left=0.045, bottom=0.085, right=0.97, top=0.965, wspace=0.2, hspace=0.09)
    plt.savefig(os.path.join(basedir, period, tag, 'distributions_'+period+'_'+signal[i]+'_syst.pdf'))
    plt.savefig(os.path.join(basedir, period, tag, 'distributions_'+period+'_'+signal[i]+'_syst.png'))
    #plt.show()

