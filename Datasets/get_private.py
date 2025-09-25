import os
import sys
import argparse

files_dir = "../AP_ZpDM_Had_R2/Datasets/Files"
sys.path.insert(0, files_dir)
from Private import *

#find . -type d -print

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version")
parser.add_argument("-p", "--period")
args = parser.parse_args()

version = args.version
period = args.period

basedir_bkg = files_dir+"/"+"bkg_"+period[-2:]+"/dti_"+period[0]+"/v"+version+"/"
if os.path.isdir(basedir_bkg) is False:
    os.makedirs(basedir_bkg)

basedir_signal = files_dir+"/"+"signal_"+period[-2:]+"/dti_"+period[0]+"/v"+version+"/"
if os.path.isdir(basedir_signal) is False:
    os.makedirs(basedir_signal)

#==================================================================================================
for i in range(len(datasets[period])):
    if datasets[period][i][0][:6] == "Signal":
        file_out = basedir_signal + datasets[period][i][0] + ".txt"
    else:
        file_out = basedir_bkg + datasets[period][i][0] + ".txt"

    files_list = os.listdir(datasets[period][i][1])

    with open(file_out, "w") as out:
        for input_file in files_list:
            out.write(os.path.join(datasets[period][i][1],input_file+"\n"))

