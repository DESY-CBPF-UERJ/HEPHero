import os
import pandas as pd
from tqdm import tqdm
import json
import sys
import h5py
import argparse
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

#==================================================================================================
def __generate_cutflow(period, basedir, datasets_path, signal_ref=None):

    #Combine cutflow file for each event process for each job directory and produce general cutflow

    #Args:
    #    basedir (str): Path to analysis root folder
    #    period (str): Jobs period used in anafile
    #    samples (dict): Dictionary mapping each event flavour to jobs directories

    samples = get_samples( basedir, period )

    cutflow_filepath = os.path.join(datasets_path, "cutflow_XX.txt")
    cutflow_file = open(cutflow_filepath, "w")

    if signal_ref is not None:
        signal_tag = signal_ref
    else:
        signal_tag = "all_signals"

    plot_n = 1
    plot_control = 0
    cut_val_signal = []

    for datasets in tqdm(samples.keys()):
        cutflow_file.write("------------------------------------------------------------------------------------"+"\n")
        cutflow_file.write("Cutflow from " + datasets + ":"+"\n")
        cutflow_file.write("------------------------------------------------------------------------------------"+"\n")
        plotted = False
        control = 0
        DATA_LUMI = 0
        PROC_XSEC = 0
        SUM_GEN_WGT = 0
        for dataset in samples[datasets]:
            dataset_name = dataset.split("_files_")[0]
            dataset_period = dataset_name.split("_")[-2]+"_"+dataset_name.split("_")[-1]
            if (dataset_period == period):
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

        if plot_control == 0:
            fig1 = plt.figure(figsize=(15,8))
            ax = plt.subplot(1,1,1)
            if len(cut_val_signal) > 0:
                plt.plot(cut_val_signal, label=signal_ref, dashes=[6, 2])
                plot_control += 1

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
            cutflow_file.write("Sum of genWeights = " + str(SUM_GEN_WGT)+"\n")
            cutflow_file.write("------------------------------------------------------------------------------------"+"\n")
            cutflow_file.write('Cutflow               Selected Events      Stat. Error         Efficiency (%)'+"\n")
            for i in range(len(cut_name)):
                cutflow_file.write("|" + cut_name[i].ljust(17) + "%18.6f %16.6f %19.4f" % (cut_val[i], cut_unc[i], (cut_val[i]*100)/SUM_GEN_WGT)+"\n")
            cutflow_file.write(""+"\n")
            cutflow_file.write(""+"\n")

            if signal_ref is not None:
                if datasets == signal_ref:
                    plt.plot(cut_val, label=datasets, dashes=[6, 2])
                    cut_val_signal = cut_val.copy()
                    plotted = True
                elif datasets[:4] != "Data" and datasets[:6] != "Signal":
                    plt.plot(cut_val, label=datasets)
                    plotted = True
            else:
                if datasets[:6] == "Signal":
                    plt.plot(cut_val, label=datasets)
                    plotted = True

        if plot_control == 7:
            ax.set_xlabel("Selection", size=14, horizontalalignment='right', x=1.0)
            ax.set_ylabel("Events", size=14, horizontalalignment='right', y=1.0)

            ax.tick_params(which='major', length=8)
            ax.tick_params(which='minor', length=4)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['top'].set_linewidth(1)
            ax.spines['left'].set_linewidth(1)
            ax.spines['right'].set_linewidth(1)
            ax.margins(x=0)
            plt.yscale('log')
            ax.grid(which='major', axis='x', linewidth=0.2, linestyle='-', color='0.75')
            ax.grid(which='major', axis='y', linewidth=0.2, linestyle='-', color='0.75')
            ax.set_ylim([1.e-1,5.e7])
            ax.legend(numpoints=1, ncol=5, prop={'size': 10.5}, frameon=False, loc='upper right')

            plt.xticks(range(len(cut_name)), cut_name, rotation = 25, ha="right")

            plt.subplots_adjust(left=0.07, bottom=0.17, right=0.98, top=0.95, wspace=0.25, hspace=0.0)

            cutflow_plot_path = os.path.join(datasets_path, "cutflow_" + period + "_" + signal_tag + "_" + str(plot_n) + ".png")
            plt.savefig(cutflow_plot_path, transparent=False, dpi=400)

            cutflow_plot_path = os.path.join(datasets_path, "cutflow_" + period + "_" + signal_tag + "_" + str(plot_n) + ".pdf")
            plt.savefig(cutflow_plot_path, transparent=False)

            plot_control = 0
            plot_n += 1
        elif plotted:
            plot_control += 1


    ax.set_xlabel("Selection", size=14, horizontalalignment='right', x=1.0)
    ax.set_ylabel("Events", size=14, horizontalalignment='right', y=1.0)

    ax.tick_params(which='major', length=8)
    ax.tick_params(which='minor', length=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.margins(x=0)
    plt.yscale('log')
    ax.grid(which='major', axis='x', linewidth=0.2, linestyle='-', color='0.75')
    ax.grid(which='major', axis='y', linewidth=0.2, linestyle='-', color='0.75')
    ax.set_ylim([1.e-1,5.e7])
    ax.legend(numpoints=1, ncol=5, prop={'size': 10.5}, frameon=False, loc='upper right')

    plt.xticks(range(len(cut_name)), cut_name, rotation = 25, ha="right")

    plt.subplots_adjust(left=0.07, bottom=0.17, right=0.98, top=0.95, wspace=0.25, hspace=0.0)


    real_cutflow_filepath = os.path.join(datasets_path, "cutflow_" + period + "_" + signal_tag + ".txt")

    cutflow_plot_path = os.path.join(datasets_path, "cutflow_" + period + "_" + signal_tag + "_" + str(plot_n) + ".png")
    plt.savefig(cutflow_plot_path, transparent=False, dpi=400)

    cutflow_plot_path = os.path.join(datasets_path, "cutflow_" + period + "_" + signal_tag + "_" + str(plot_n) + ".pdf")
    plt.savefig(cutflow_plot_path, transparent=False)

    os.system("mv " + cutflow_filepath + " " + real_cutflow_filepath)

    cutflow_file.close()


#==================================================================================================
def __join_cutflows(periods, basedir, datasets_path, signal_ref=None):

    cutflow_filepath = os.path.join(datasets_path, "cutflow_XX.txt")
    cutflow_file = open(cutflow_filepath, "w")

    if signal_ref is not None:
        signal_tag = signal_ref
    else:
        signal_tag = "all_signals"

    sample_names = []
    for period in periods:
        samples_i = get_samples( basedir, period )
        sample_names = sample_names + [name for name in samples_i.keys() if name not in sample_names]

    bad_datasets = []
    for datasets in tqdm(sample_names):
        cutflow_file.write("------------------------------------------------------------------------------------"+"\n")
        cutflow_file.write("Cutflow from " + datasets + ":"+"\n")
        cutflow_file.write("------------------------------------------------------------------------------------"+"\n")
        control = 0
        ds_control = 0
        SUM_GEN_WGT = 0
        for period in periods:
            cutflow = os.path.join(datasets_path, "cutflow_" + period + "_" + signal_tag + ".txt")
            cut_name = []
            cut_val_i = []
            cut_unc_i = []
            if os.path.isfile(cutflow):
                with open(cutflow) as f:
                    for line in f:
                        if datasets+":" in line:
                            ds_control = 1
                        elif ds_control == 1 and line[:12] == "Cutflow from":
                            ds_control = 0

                        if ds_control == 1:
                            if line[:17] == "Sum of genWeights" :
                                SUM_GEN_WGT += float(line.split("=")[1])
                            if line[0] == "|" :
                                line_info = line.split()
                                cut_name.append(line_info[0][1:])
                                cut_val_i.append(float(line_info[1]))
                                cut_unc_i.append(float(line_info[2])**2)
                if len(cut_val_i) > 0:
                    if control == 0:
                        cut_val = np.array(cut_val_i)
                        cut_unc = np.array(cut_unc_i)
                        control = 1
                    else:
                        cut_val = cut_val + np.array(cut_val_i)
                        cut_unc = cut_unc + np.array(cut_unc_i)
                else:
                    bad_datasets.append(datasets+"_"+period)

        if control == 1:
            cut_unc = np.sqrt(cut_unc)
            cutflow_file.write("Sum of genWeights = " + str(SUM_GEN_WGT)+"\n")
            cutflow_file.write("------------------------------------------------------------------------------------"+"\n")
            cutflow_file.write('Cutflow               Selected Events      Stat. Error         Efficiency (%)'+"\n")
            for i in range(len(cut_name)):
                cutflow_file.write(cut_name[i].ljust(17) + "%18.6f %16.6f %19.4f" % (cut_val[i], cut_unc[i], (cut_val[i]*100)/SUM_GEN_WGT)+"\n")
            cutflow_file.write(""+"\n")
            cutflow_file.write(""+"\n")


    real_cutflow_filepath = os.path.join(datasets_path, "cutflow_all_periods_" + signal_tag + ".txt")
    os.system("mv " + cutflow_filepath + " " + real_cutflow_filepath)

    cutflow_file.close()

    return bad_datasets


#======SETUP=======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--selection")
parser.add_argument("-p", "--period")
parser.add_argument("-ps", "--periods")
parser.set_defaults(periods=None)
parser.add_argument("-r", "--signal_ref")
parser.set_defaults(signal_ref=None)
args = parser.parse_args()

with open('analysis.txt') as f:
    analysis = f.readline()

sys.path.insert(0, '../'+analysis+'/Datasets')
from Samples import *

if args.periods is not None:
    periods = args.periods.split(",")
else:
    periods = [args.period]


outpath = os.environ.get("HEP_OUTPATH")
machines = os.environ.get("MACHINES")
user = os.environ.get("USER")
storage_redirector = os.environ.get("STORAGE_REDIRECTOR")
basedir = os.path.join(outpath, analysis, args.selection)

datasets_path = os.path.join(basedir, "datasets")

if machines == "UERJ":
    basedir = os.path.join("/cms/store/user/", user, "output", analysis, args.selection)


if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)


jobs_file_name = os.path.join(basedir, "jobs.txt")
if(not os.path.isfile(jobs_file_name)):
    print('Missing configuration files, execute the runSelection.py using the flag "fix" as in the example below, and then try again!')
    print('python runSelection.py -j 0 --fix')
    sys.exit()



for period in periods:
    print('')
    print('Analysis = ' + analysis)
    print('Selection = ' + args.selection)
    print('Period = ' + period)
    print('Outpath = ' + basedir)

    __generate_cutflow(period, basedir, datasets_path, signal_ref=args.signal_ref)


if args.periods is not None:
    print(" ")
    bad_datasets = __join_cutflows(periods, basedir, datasets_path, signal_ref=args.signal_ref)

    for dataset in bad_datasets:
        print("The dataset", dataset, "is empty!")

if machines == "UERJ":
    cp_command = "xrdcp -rf " + datasets_path + " root://"+storage_redirector+"//store/user/"+user+"/output/"+analysis+"/"+args.selection
    os.system(cp_command)
    rm_command = "rm -rf " + datasets_path
    os.system(rm_command)
