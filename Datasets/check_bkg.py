import os
import sys
import argparse

files_dir = "../AP_ZpDM_Had_R2/Datasets/Files"
sys.path.insert(0, files_dir)
from Bkg import *

# List of datasets generated with the dasgoclient command.
# Example: dasgoclient --limit 0 --query 'dataset dataset=/QCD*/*22*v12*/NANOAODSIM'

#======SETUP=======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tag")
args = parser.parse_args()

campaign = "*"+args.tag+"*/NANOAODSIM"

datasets = [ds[1][1:-1] for ds in datasets]

print("#############################################################################################")
for i in range(len(datasets)):
    command = "dasgoclient --limit 0 --query 'dataset dataset=/" + datasets[i] + "*/" + campaign + "'"
    print(">>>> " + command)
    os.system(command)
    print(" ")
