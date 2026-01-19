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
basedir = "/home/gcorreia/cernbox/HEP/HHDM/SysTest/datasets"
period = '17'
luminosity = '41.5'

ds = data.read_files(basedir, period)


data.join_datasets(ds, "ST", [
    "ST_tW_antitop", 
    "ST_tW_top", 
    #"ST_s-channel",
    #"ST_t-channel_top", 
    #"ST_t-channel_antitop",
    ])

data.join_datasets(ds, "TT", [
    "TTTo2L2Nu", 
    "TTToSemiLeptonic",
    ])

data.join_datasets(ds, "DYJetsToLL", [
    "DYJetsToLL_Pt-Inclusive",
    ])

data.join_datasets(ds, "Residual", [
    "WZZ", 
    "WWZ", 
    "ZZZ", 
    "WWW", 
    "ZGToLLG", 
    "TTGamma", 
    "TTGJets", 
    "WW",
    #"WGToLNuG", 
    #"TTZToQQ", 
    #"TTZToNuNu", 
    #"TWZToLL_thad_Wlept", 
    #"TWZToLL_tlept_Whad", 
    #"TWZToLL_tlept_Wlept", 
    #"WJetsToLNu", 
    #"TTWZ", 
    #"TTZZ",
    ])

data.join_datasets(ds, "Data", [ # It changes for each year
    "Data_B", 
    "Data_C", 
    "Data_D", 
    "Data_E", 
    "Data_F",
    ])


def signal_label(param_0, param_1):
    label = r'$m_H=$' + str(param_0) + r', $m_\mathit{a}=$' + str(param_1)
    return label

# %%
#==================================================================================================
# Plot
#==================================================================================================
fig1 = plt.figure(figsize=(20,7.5))
grid = [2, 3]
gs1 = gs.GridSpec(grid[0], grid[1], height_ratios=[4, 1])

df = ds.copy()
for key in df.keys():
    dataset = df[key]
    df[key] = dataset[(dataset["RegionID"] == 2)]

colors = ['gainsboro', 'orchid', 'limegreen', 'red', 'skyblue', 'darkgoldenrod']
labels = [r'Residual SM', r'$WZ$', r'$ZZ$', 'Single top', r'$t\bar{t}$', 'Drell-Yan']
dataframes = [df["Residual"], df["WZ"], df["ZZ"], df["ST"], df["TT"], df["DYJetsToLL"]]
dataframes, labels, colors, sizes = data.order_datasets(dataframes, labels, colors)


#=================================================================================================================
N = 1
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "MET_pt"
bins = [40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400, 500, 700]
ybkg, errbkg = ana.stacked_plot( ax1, var, dataframes, labels, colors, weight="evtWeight", bins=bins )  # Produce the stacked plot
#print(ybkg)
ydata, errdata = ana.data_plot( ax1, var, df["Data"], bins=bins )
#print(ydata)
ana.labels(ax1, ylabel="Events")  # Set up the label names
ana.style(ax1, lumi=luminosity, year=period[-2:], ylog=True, legend_ncol=4, ylim=[1.e-1,1.e5]) # Set up the plot style and information on top

#==================================================
ax2 = plt.subplot(ana.position(gs1,grid,N,2), sharex=ax1)  # Positioning at subplot 2 of the plot number 2
#==================================================
ana.ratio_plot( ax2, ydata, errdata, ybkg, errbkg, bins=bins)
ana.labels(ax2, xlabel=r"$E_T^\mathrm{miss}\,[\mathrm{GeV}]$", ylabel="Data / Bkg.")  # Set up the label names
ana.style(ax2, ylim=[0., 2], yticks=[0, 0.5, 1, 1.5, 2], xgrid=True, ygrid=True)


#=================================================================================================================
N = 2
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "Nbjets"
bins = np.linspace(0,10,11)
ybkg, errbkg = ana.stacked_plot( ax1, var, dataframes, labels, colors, weight="evtWeight", bins=bins )  # Produce the stacked plot
#print(ybkg)
ydata, errdata = ana.data_plot( ax1, var, df["Data"], bins=bins )
#print(ydata)
ana.labels(ax1, ylabel="Events")  # Set up the label names
ana.style(ax1, lumi=luminosity, year=period[-2:], ylog=True, legend_ncol=4, ylim=[1.e-1,1.e5]) # Set up the plot style and information on top

#==================================================
ax2 = plt.subplot(ana.position(gs1,grid,N,2), sharex=ax1)  # Positioning at subplot 2 of the plot number 2
#==================================================
ana.ratio_plot( ax2, ydata, errdata, ybkg, errbkg, bins=bins)
ana.labels(ax2, xlabel=r"$N_\mathrm{b\,jets}$", ylabel="Data / Bkg.")  # Set up the label names
ana.style(ax2, ylim=[0., 2], yticks=[0, 0.5, 1, 1.5, 2], xgrid=True, ygrid=True)


#=================================================================================================================
N = 3
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "LeadingLep_pt"
bins = [40, 55, 70, 85, 100, 115, 130, 145, 160, 175, 190, 205, 220, 250, 400]
ybkg, errbkg = ana.stacked_plot( ax1, var, dataframes, labels, colors, weight="evtWeight", bins=bins )  # Produce the stacked plot
#print(ybkg)
ydata, errdata = ana.data_plot( ax1, var, df["Data"], bins=bins )
#print(ydata)
ana.labels(ax1, ylabel="Events")  # Set up the label names
ana.style(ax1, lumi=luminosity, year=period[-2:], ylog=True, legend_ncol=4, ylim=[1.e-1,1.e5]) # Set up the plot style and information on top

#==================================================
ax2 = plt.subplot(ana.position(gs1,grid,N,2), sharex=ax1)  # Positioning at subplot 2 of the plot number 2
#==================================================
ana.ratio_plot( ax2, ydata, errdata, ybkg, errbkg, bins=bins)
ana.labels(ax2, xlabel=r"$\mathrm{Leading\, Lepton\,} p_T\,[\mathrm{GeV}]$", ylabel="Data / Bkg.")  # Set up the label names
ana.style(ax2, ylim=[0., 2], yticks=[0, 0.5, 1, 1.5, 2], xgrid=True, ygrid=True)


#=================================================================================================================
# Make final setup, save and show plots
#=================================================================================================================
plt.subplots_adjust(left=0.045, bottom=0.085, right=0.97, top=0.965, wspace=0.2, hspace=0.09)
plt.savefig('distributions_'+period+'.pdf')
plt.savefig('distributions_'+period+'.png')
plt.show()

