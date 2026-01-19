import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import hepherolib.data as data
import hepherolib.analysis as ana
ana.start()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# This is not a complete code. You need to generate the datasets and the lists below: 
# dataframes - list of pandas dataframes
# labels - list of strings setting the labels
# colors - list of strings setting the colors
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#=================================================================================================================
# Set up the figure and the subplots grid
#=================================================================================================================
fig1 = plt.figure(figsize=(20,6))
grid = [2, 3]
gs1 = gs.GridSpec(grid[0], grid[1], height_ratios=[4, 1])


#=================================================================================================================
N = 1
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "LeadingLep_pt"
bins = np.linspace(0,600,61)
yratio, ye_below, ye_above = ana.efficiency_plot( ax1, var, df_400_100, "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", label="binomial", color='black', bins=bins, histograms=True, uncertainty="binomial" )
ana.labels(ax1, xlabel=r"$\mathrm{Leading\ } p_T^l\ [\mathrm{GeV}]$", ylabel=r'Efficiency')  # Set up the label names
ana.style(ax1, lumi=35.9, year=2016, legend_ncol=1, legend_loc='center right') # Set up the plot style and information on top


#=================================================================================================================
N = 2
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "LeadingLep_pt"
bins = np.linspace(0,600,61)
yratio, ye_below, ye_above = ana.efficiency_plot( ax1, var, df_400_100, "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", label="bayesian", color='black', bins=bins, histograms=True )
ana.labels(ax1, xlabel=r"$\mathrm{Leading\ } p_T^l\ [\mathrm{GeV}]$", ylabel=r'Efficiency')  # Set up the label names
ana.style(ax1, lumi=35.9, year=2016, legend_ncol=1, legend_loc='center right') # Set up the plot style and information on top


#=================================================================================================================
N = 3
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "LeadingLep_pt"
bins = np.linspace(0,600,61)
yratio, ye_below, ye_above = ana.efficiency_plot( ax1, var, df_400_100, "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", label="HLT_Ele23_Ele12", color='blue', bins=bins )
yratio, ye_below, ye_above = ana.efficiency_plot( ax1, var, df_400_100, "HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_MW", label="HLT_DoubleEle33", color='red', bins=bins )
ana.labels(ax1, xlabel=r"$\mathrm{Leading\ } p_T^l\ [\mathrm{GeV}]$", ylabel=r'Efficiency')  # Set up the label names
ana.style(ax1, lumi=35.9, year=2016, legend_ncol=1, legend_loc='lower right') # Set up the plot style and information on top



#=================================================================================================================
# Make final setup, save and show plots
#=================================================================================================================
plt.subplots_adjust(left=0.055, bottom=0.115, right=0.96, top=0.95, wspace=0.35, hspace=0.0)
plt.savefig('efficiency_plots.png')
plt.savefig('efficiency_plots.pdf')
plt.show()



