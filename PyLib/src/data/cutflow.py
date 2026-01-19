import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ..analysis import style, labels



def generate_cutflow(basedir, period, samples, num_plots=6, lumi=35.9, year=2016, outpath=os.getcwd()):
    """
    Combine cutflow file for each event process for each job directory and produce general cutflow

    Args:
        basedir (str): Path to analysis root folder
        period (str): Jobs period used in anafile
        samples (dict): Dictionary mapping each event flavour to jobs directories
    """
    cutflow_filepath = os.path.join(outpath, "cutflow_XX.txt")
    cutflow_file = open(cutflow_filepath, "w")

    fig1 = plt.figure(figsize=(25,8))
    plot_control = 0
    plot_n = 1
    NumPlots = num_plots

    has_tag = False  # Remove if CMS join 2016 samples again
    for datasets in tqdm(samples.keys()):
        cutflow_file.write("------------------------------------------------------------------------------------"+"\n")
        cutflow_file.write("Cutflow from " + datasets + ":"+"\n")
        cutflow_file.write("------------------------------------------------------------------------------------"+"\n")
        ax = plt.subplot(1,NumPlots,plot_n)
        plotted = False
        control = 0
        DATA_LUMI = 0
        PROC_XSEC = 0
        SUM_GEN_WGT = 0
        for dataset in samples[datasets]:
            dataset_year = dataset.split("_files_")[0]
            dataset_year = dataset_year.split("_")[-1]
            dataset_tag = dataset.split("_"+dataset_year)[0][-3:]
            if (dataset_year == period):
                if( dataset_tag == "APV" ): 
                    has_tag = True
                cutflow = os.path.join(basedir, dataset, "cutflow.txt")
                cut_name = []
                cut_val_i = []
                cut_unc_i = []
                if os.path.isfile(cutflow):
                    with open(cutflow) as f:
                        for line in f:
                            if line[:10] == "Luminosity" :
                                DATA_LUMI = float(line.split()[1])
                            if line[:13] == "Cross section" :
                                PROC_XSEC = float(line.split()[2])
                            if line[:17] == "Sum of genWeights" :
                                SUM_GEN_WGT += float(line.split()[3])
                            if line[0] == "|" :
                                line_info = line.split()
                                cut_name.append(line_info[0][1:])
                                cut_val_i.append(float(line_info[1]))
                                cut_unc_i.append(float(line_info[2])**2)
                    if control == 0:
                        cut_val = np.array(cut_val_i)
                        cut_unc = np.array(cut_unc_i)
                        control = 1
                    else:
                        cut_val = cut_val + np.array(cut_val_i)
                        cut_unc = cut_unc + np.array(cut_unc_i)

        if control == 1:
            cut_unc = np.sqrt(cut_unc)
            if PROC_XSEC == 0:
                dataScaleWeight = 1
                SUM_GEN_WGT = -1
            else:
                dataScaleWeight = (PROC_XSEC/SUM_GEN_WGT) * DATA_LUMI
                SUM_GEN_WGT = SUM_GEN_WGT*dataScaleWeight
            cut_val = cut_val*dataScaleWeight
            cut_unc = cut_unc*dataScaleWeight
            cutflow_file.write("Data scale weight = " + str(dataScaleWeight)+"\n")
            cutflow_file.write("------------------------------------------------------------------------------------"+"\n")
            cutflow_file.write('Cutflow               Selected Events      Stat. Error         Efficiency (%)'+"\n")
            for i in range(len(cut_name)):
                cutflow_file.write(cut_name[i].ljust(17) + "%18.6f %16.6f %19.4f" % (cut_val[i], cut_unc[i], (cut_val[i]*100)/SUM_GEN_WGT)+"\n")
            cutflow_file.write(""+"\n")
            cutflow_file.write(""+"\n")


            if datasets == 'Signal_400_100' or datasets == 'Signal_1000_800':
                for i in range(NumPlots):
                    ax = plt.subplot(1,NumPlots,i+1)
                    plt.plot(cut_val, label=datasets, dashes=[6, 2])
                ax = plt.subplot(1,NumPlots,plot_n)
            elif datasets[:4] != "Data" and datasets[:6] != "Signal":
                plt.plot(cut_val, label=datasets)
                plotted = True
    
        if plot_control == 7:
            labels(ax, ylabel="Events", xlabel="Selection")
            style(ax, lumi=lumi, year=year, ylog=True, xgrid=True, ygrid=True, ylim=[1.e-1,5.e7], legend_ncol=5)
            plt.xticks(range(len(cut_name)), cut_name, rotation = 25, ha="right")
            plot_control = 0
            plot_n += 1
        elif plotted:
            plot_control += 1
    
    labels(ax, ylabel="Events", xlabel="Selection")
    style(ax, lumi=lumi, year=year, ylog=True, xgrid=True, ygrid=True, ylim=[1.e-1,5.e7], legend_ncol=5)
    plt.xticks(range(len(cut_name)), cut_name, rotation = 25, ha="right")
    plt.subplots_adjust(left=0.055, bottom=0.17, right=0.98, top=0.95, wspace=0.25, hspace=0.0)
    
    cutflow_file.close()
    if( has_tag ):
        real_cutflow_filepath = os.path.join(outpath, "cutflow_APV_" + period + ".txt")
        cutflow_plot_path = os.path.join(outpath, "cutflow_APV_" + period + ".png")
    else:
        real_cutflow_filepath = os.path.join(outpath, "cutflow_" + period + ".txt")
        cutflow_plot_path = os.path.join(outpath, "cutflow_" + period + ".png")
    
    plt.savefig(cutflow_plot_path)
    os.system("mv " + cutflow_filepath + " " + real_cutflow_filepath)    
