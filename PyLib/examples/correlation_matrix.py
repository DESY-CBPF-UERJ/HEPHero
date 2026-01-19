import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import hepherolib.data as data
import hepherolib.analysis as ana
ana.start()

basedir = "/home/gcorreia/cernbox/HEP_Project/CMS_HHDM/OUTPUT/Test"
list_basedir = os.listdir(basedir)
TreeName = 'selection'
period = '16'

#=================================================================================================================
# Read files, apply selection and associate datasets
#=================================================================================================================
datasets = data.read_files(basedir, period)

for key in datasets.keys():
    dataset = datasets[key] 
    datasets[key] = dataset[(dataset["RecoLepID"] < 1000) & (dataset["Nbjets"] > 0)]

    
df_400_100 = datasets.get("Signal_400_100")
df_TTTo2L2Nu = datasets.get("TTTo2L2Nu")





#=================================================================================================================
# Analysis
#=================================================================================================================    
variables = ["LepLep_pt", "LepLep_daltaR", "LeadingLep_pt", "MET_LepLep_Mt", "Nbjets"]
var_names = [r"$p_\mathrm{T}^\mathrm{ll}$", r"$\Delta R^\mathrm{ll}$", r"$\mathrm{leading}\,p_\mathrm{T}^\mathrm{l}$", r"$M_\mathrm{T}^\mathrm{ll,MET}$", r"$N_\mathrm{b\,jets}$"]

"""
$\mathrm{leading}\,p_\mathrm{T}^\mathrm{l}$ 
$\mathrm{trailing}\,p_\mathrm{T}^\mathrm{l}$
$\mathrm{leading}\,\eta^\mathrm{l}$
$\mathrm{trailing}\,\eta^\mathrm{l}$
$p_\mathrm{T}^\mathrm{ll}$
$\Delta M^\mathrm{ll}$
$\Delta R^\mathrm{ll}$
$\Delta \phi^\mathrm{ll,MET}$
$M_\mathrm{T}^\mathrm{ll,MET}$
$E_\mathrm{T}^\mathrm{miss}$
$N_\mathrm{b\,jets}$
"""

df_signal = df_400_100
df_bkg = df_TTTo2L2Nu

      
fig = plt.figure(figsize=(18,10))

ax = plt.subplot(1,2,1)
ana.cov_matrix_plot(ax, df_signal, variables, weight="evtWeight", title="Correlation matrix - Signal", title_size=22, text_size=18, var_names=var_names)

ax = plt.subplot(1,2,2)
matrix = ana.cov_matrix_plot(ax, df_bkg, variables, weight="evtWeight", title="Correlation matrix - Background")

print(matrix)
        
fig.tight_layout()  
plt.savefig('correlation.png', fancybox=True)
plt.savefig('correlation.pdf')









