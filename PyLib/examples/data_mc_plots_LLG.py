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
basedir = "/home/gcorreia/cernbox/HEP/LLG/SysTest/datasets"
period = '18'
luminosity = '59.8'

ds = data.read_files(basedir, period)

data.join_datasets(ds, "ST", [
    "ST_tW_antitop", 
    "ST_tW_top", 
    #"ST_s-channel",
    #"ST_t-channel_top", 
    #"ST_t-channel_antitop",
    ])

data.join_datasets(ds, "WJets", [
    "WJetsToLNu_Pt-100To250",
    "WJetsToLNu_Pt-250To400",
    "WJetsToLNu_Pt-400To600",
    "WJetsToLNu_Pt-600ToInf",
    ])

data.join_datasets(ds, "QCD", [
    #"QCD_HT100to200",
    #"QCD_HT200to300",
    "QCD_HT300to500",
    "QCD_HT500to700",
    "QCD_HT700to1000",
    "QCD_HT1000to1500",
    "QCD_HT1500to2000",
    "QCD_HT2000toInf",
    ])

data.join_datasets(ds, "ZJets", [
    "ZJetsToNuNu_HT-100To200",
    "ZJetsToNuNu_HT-200To400",
    "ZJetsToNuNu_HT-400To600",
    "ZJetsToNuNu_HT-600To800",
    "ZJetsToNuNu_HT-800To1200",
    "ZJetsToNuNu_HT-1200To2500",
    "ZJetsToNuNu_HT-2500ToInf",
    ])

data.join_datasets(ds, "Residual", [
    "WZ", 
    "WW", 
    "ZZ", 
    #"GJets_DR-0p4_HT-100To200",
    "GJets_DR-0p4_HT-200To400",
    "GJets_DR-0p4_HT-400To600",
    "GJets_DR-0p4_HT-600ToInf",
    ])

data.join_datasets(ds, "Data", [ # It changes for each year
    "Data_A", 
    "Data_B", 
    "Data_C",
    "Data_D",
    ])




def signal_label(param_0, param_1):
    label = r'$m_{\tilde{g}}=$' + str(param_0) + r', $\Delta m=$' + str(param_1)
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
    df[key] = dataset[(dataset["RegionID"] == 1)]

colors = ['gainsboro', 'orchid', 'skyblue', 'orange', 'red', 'limegreen']
labels = [r'Residual SM', r'$W(l\nu)$+jets', r'$t\bar{t}$+jets', 'Single top', 'QCD', r'$Z(\nu\nu)$+jets']
dataframes = [df["Residual"], df["WJets"], df["TTJets"], df["ST"], df['QCD'], df["ZJets"]]

dataframes, labels, colors, sizes = data.order_datasets(dataframes, labels, colors)


#=================================================================================================================
N = 1
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "Njets30"
bins = np.linspace(0,12,13)
ybkg, errbkg = ana.stacked_plot( ax1, var, dataframes, labels, colors, weight="evtWeight", bins=bins )  # Produce the stacked plot
#print(ybkg)
ydata, errdata = ana.data_plot( ax1, var, df["Data"], bins=bins )
#print(ydata)
ana.labels(ax1, ylabel="Events")  # Set up the label names
ana.style(ax1, lumi=luminosity, year=period[-2:], ylog=True, legend_ncol=4, ylim=[1.e-1,1.e6]) # Set up the plot style and information on top

#==================================================
ax2 = plt.subplot(ana.position(gs1,grid,N,2), sharex=ax1)  # Positioning at subplot 2 of the plot number 2
#==================================================
ana.ratio_plot( ax2, ydata, errdata, ybkg, errbkg, bins=bins)
ana.labels(ax2, xlabel=r"$N_\mathrm{jets30}$", ylabel="Data / Bkg.")  # Set up the label names
ana.style(ax2, ylim=[0., 2], yticks=[0, 0.5, 1, 1.5, 2], xgrid=True, ygrid=True)


#=================================================================================================================
N = 2
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "MET_pt"
bins = np.linspace(200,1200,26)
ybkg, errbkg = ana.stacked_plot( ax1, var, dataframes, labels, colors, weight="evtWeight", bins=bins )  # Produce the stacked plot
#print(ybkg)
ydata, errdata = ana.data_plot( ax1, var, df["Data"], bins=bins )
#print(ydata)
ana.labels(ax1, ylabel="Events")  # Set up the label names
ana.style(ax1, lumi=luminosity, year=period[-2:], ylog=True, legend_ncol=4, ylim=[1.e-1,1.e6]) # Set up the plot style and information on top

#==================================================
ax2 = plt.subplot(ana.position(gs1,grid,N,2), sharex=ax1)  # Positioning at subplot 2 of the plot number 2
#==================================================
ana.ratio_plot( ax2, ydata, errdata, ybkg, errbkg, bins=bins)
ana.labels(ax2, xlabel=r"$E_\mathrm{T}^{miss}\,[\mathrm{GeV}]$", ylabel="Data / Bkg.")  # Set up the label names
ana.style(ax2, ylim=[0., 2], yticks=[0, 0.5, 1, 1.5, 2], xgrid=True, ygrid=True)


#=================================================================================================================
N = 3
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "HT30"
bins = np.linspace(300,1300,26)
ybkg, errbkg = ana.stacked_plot( ax1, var, dataframes, labels, colors, weight="evtWeight", bins=bins )  # Produce the stacked plot
#print(ybkg)
ydata, errdata = ana.data_plot( ax1, var, df["Data"], bins=bins )
#print(ydata)
ana.labels(ax1, ylabel="Events")  # Set up the label names
ana.style(ax1, lumi=luminosity, year=period[-2:], ylog=True, legend_ncol=4, ylim=[1.e-1,1.e6]) # Set up the plot style and information on top

#==================================================
ax2 = plt.subplot(ana.position(gs1,grid,N,2), sharex=ax1)  # Positioning at subplot 2 of the plot number 2
#==================================================
ana.ratio_plot( ax2, ydata, errdata, ybkg, errbkg, bins=bins)
ana.labels(ax2, xlabel=r"$H_\mathrm{T}\,[\mathrm{GeV}]$", ylabel="Data / Bkg.")  # Set up the label names
ana.style(ax2, ylim=[0., 2], yticks=[0, 0.5, 1, 1.5, 2], xgrid=True, ygrid=True)


#=================================================================================================================
# Make final setup, save and show plots
#=================================================================================================================
plt.subplots_adjust(left=0.045, bottom=0.085, right=0.97, top=0.965, wspace=0.2, hspace=0.09)
plt.savefig('distributions_'+period+'.pdf')
plt.savefig('distributions_'+period+'.png')
plt.show()


