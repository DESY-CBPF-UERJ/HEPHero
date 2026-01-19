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
var = "MET_LepLep_Mt"
bins = np.linspace(150,450,21)
ana.step_plot( ax1, var, df_1000_100, label=signal_label(1000,100), color='blue', weight="evtWeight", bins=bins )
ana.step_plot( ax1, var, df_1000_800, label=signal_label(1000,800), color='turquoise', weight="evtWeight", bins=bins )
ana.step_plot( ax1, var, df_400_100, label=signal_label(400,100), color='slategray', weight="evtWeight", bins=bins )
ybkg, errbkg = ana.stacked_plot( ax1, var, dataframes, labels, colors, weight="evtWeight", bins=bins )  # Produce the stacked plot
ydata, errdata = ana.data_plot( ax1, var, df_DATA, bins=bins )
ana.labels(ax1, ylabel="Events")  # Set up the label names
ana.style(ax1, lumi=35.9, year=2016, ylog=True, legend_ncol=2, ylim=[1.e-1,1.e6], xticklabels=False) # Set up the plot style and information on top

#==================================================
ax2 = plt.subplot(ana.position(gs1,grid,N,2), sharex=ax1)  # Positioning at subplot 2 of the plot number 2
#==================================================
ana.ratio_plot( ax2, ydata, errdata, ybkg, errbkg, bins=bins)
ana.labels(ax2, xlabel=r"$M_T^{ll, \mathrm{MET}}$", ylabel="Data / Bkg.")  # Set up the label names
ana.style(ax2, ylim=[0., 2], yticks=[0, 0.5, 1, 1.5, 2], xgrid=True, ygrid=True) 



#=================================================================================================================
N = 2
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "MET_LepLep_deltaPhi"
bins = np.linspace(0.8,3.2,81)
ana.stacked_plot( ax1, var, dataframes, labels, colors, weight="evtWeight", bins=bins )  # Produce the stacked plot
ana.step_plot( ax1, var, df_1000_100, label=signal_label(1000,100), color='blue', weight="evtWeight", bins=bins )
ana.step_plot( ax1, var, df_1000_800, label=signal_label(1000,800), color='turquoise', weight="evtWeight", bins=bins )
ana.step_plot( ax1, var, df_400_100, label=signal_label(400,100), color='slategray', weight="evtWeight", bins=bins )
ana.labels(ax1, ylabel="Events")  # Set up the label names
ana.style(ax1, lumi=35.9, year=2016, ylog=True, legend_ncol=3, ylim=[1.e-2,1.e6], xticklabels=False) # Set up the plot style and information on top

#==================================================
ax2 = plt.subplot(ana.position(gs1,grid,N,2), sharex=ax1)  # Positioning at subplot 2 of the plot number 2
#==================================================
ctr = ana.control( var, [df_1000_800], dataframes, weight="evtWeight", bins=np.linspace(0.8,3.2,1001) )
#ctr.purity_plot()
ctr.signal_eff_plot(label='Signal_1000_800 efficiency')
ctr.bkg_eff_plot()
ana.labels(ax2, xlabel=r"$\Delta \phi^{ll, \mathrm{MET}}$", ylabel="Control")  # Set up the label names
ana.style(ax2, ylim=[0., 1.1], yticks=[0., 0.2, 0.4, 0.6, 0.8, 1.], xgrid=True, ygrid=True) 


for key in datasets.keys():
    dataset = datasets[key] 
    datasets[key] = dataset[(dataset["MET_LepLep_Mt"] < 410)]


#=================================================================================================================
N = 3
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "MET_LepLep_Mt"
bins = np.linspace(150,450,21)
ysgn1, errsgn1 = ana.step_plot( ax1, var, df_1000_800, label=r'$t\bar{t}$ CR', color='blue', weight="evtWeight", bins=bins, error=True, normalize=True )
ysgn2, errsgn2 = ana.step_plot( ax1, var, df_400_100, label=r'$t\bar{t}$ SR', color='red', weight="evtWeight", bins=bins, error=True, normalize=True )
ana.labels(ax1, ylabel="Events")  # Set up the label names
ana.style(ax1, lumi=35.9, year=2016, legend_ncol=1, xticklabels=False) # Set up the plot style and information on top

#==================================================
ax2 = plt.subplot(ana.position(gs1,grid,N,2), sharex=ax1)  # Positioning at subplot 2 of the plot number 2
#==================================================
ana.ratio_plot( ax2, ysgn1, errsgn1, ysgn2, errsgn2, bins=bins, numerator="mc", color='blue')
ana.labels(ax2, xlabel=r"$M_T^{ll, \mathrm{MET}}$", ylabel=r'CR / SR')  # Set up the label names
ana.style(ax2, ylim=[0., 4], yticks=[0., 1, 2, 3, 4], xgrid=True, ygrid=True) 


#=================================================================================================================
# Make final setup, save and show plots
#=================================================================================================================
plt.subplots_adjust(left=0.055, bottom=0.115, right=0.96, top=0.95, wspace=0.35, hspace=0.0)
plt.savefig('general_plots.png')
plt.savefig('general_plots.pdf')
plt.show()





