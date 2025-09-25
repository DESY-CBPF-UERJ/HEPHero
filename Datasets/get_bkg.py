import os
import sys
import argparse

files_dir = "../AP_ZpDM_Had_R2/Datasets/Files"
sys.path.insert(0, files_dir)
from Bkg import *

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version")
parser.add_argument("-p", "--period")
args = parser.parse_args()

version = args.version
period = args.period

basedir = files_dir+"/"+"bkg_"+period[-2:]+"/dti_"+period[0]+"/v"+version+"/"
if os.path.isdir(basedir) is False:
    os.makedirs(basedir)

#==================================================================================================
campaign_key = version+"_"+period
if campaign_key in campaigns:
    for i in range(len(datasets)):
        file_name = basedir + datasets[i][0]+".txt"

        # Main dataset
        for k in range(9):
            ds_name = datasets[i][1]+campaigns[campaign_key] + "-v" + str(k+1) + "/NANOAODSIM"
            if k == 0:
                command = "dasgoclient --limit 0 --query 'dataset dataset=" + ds_name + "' > " + "temp.txt"
            else:
                command = "dasgoclient --limit 0 --query 'dataset dataset=" + ds_name + "' >> " + "temp.txt"
            os.system(command)
        has_dataset = False
        k_lines = []
        with open("temp.txt", "r") as file:
            for line in file:
                has_dataset = True
                k_lines.append(line[0:-1])
        os.system("rm temp.txt")
        if has_dataset:
            k_value = 9
            for dataset_name in reversed(k_lines):
                command = "dasgoclient --limit 0 --query 'file dataset=" + dataset_name + "' > " + file_name
                os.system(command)
                NumLines = sum(1 for line in open(file_name))
                if NumLines > 0:
                    print(dataset_name)
                    break
                k_value = k_value - 1
        else:
            open(file_name, 'a').close()
            print(datasets[i][1]+campaigns[campaign_key] + " is not available!")

        # Extension
        ds_name = datasets[i][1]+campaigns[campaign_key] + "_ext*-v" + str(k_value) + "/NANOAODSIM"
        command = "dasgoclient --limit 0 --query 'dataset dataset=" + ds_name + "' > " + "temp.txt"
        os.system(command)
        has_dataset = False
        ext_lines = []
        with open("temp.txt", "r") as file:
            for line in file:
                has_dataset = True
                ext_lines.append(line[0:-1])
        os.system("rm temp.txt")
        if has_dataset:
            for dataset_name in ext_lines:
                command = "dasgoclient --limit 0 --query 'file dataset=" + dataset_name + "' >> " + file_name
                os.system(command)
                print(dataset_name)
else:
    print("There is no campaign set for this period and NANOAOD version!")
    sys.exit()
