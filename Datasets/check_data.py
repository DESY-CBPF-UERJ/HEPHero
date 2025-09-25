import os
import sys
import argparse

files_dir = "../AP_Template_R3/Datasets/Files"
sys.path.insert(0, files_dir)
from Data import *

# List of datasets generated with the dasgoclient command.
# Example: dasgoclient --limit 0 --query 'dataset dataset=/MET*/*22*/NANOAOD'

#======SETUP=======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year")
parser.add_argument("-t", "--tag")
args = parser.parse_args()

campaign = "*"+args.tag+"*/NANOAOD"

datasets = [ds[1][1:-1] for ds in datasets[args.year]]

print("#############################################################################################")
for i in range(len(datasets)):
    command = "dasgoclient --limit 0 --query 'dataset dataset=/" + datasets[i] + "/" + campaign + "'"
    print(">>>> " + command)
    os.system(command)
    print(" ")


