import os
import sys
import argparse

# List of datasets generated with the command below (example for MuonEG). 
# dasgoclient --limit 0 --query 'dataset dataset=/MuonEG/*UL2016_MiniAODv1_NanoAODv2*/NANOAOD'

#======SETUP=======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version")
parser.set_defaults(version=9)
args = parser.parse_args()

if args.version != "9":
    print("Please, enter a valid version: 9")
    sys.exit()

if args.version == "9":
    campaign = "UL2017_MiniAODv2_NanoAODv9"
    campaignSingleMu = "UL2017_MiniAODv2_NanoAODv9_GT36"


basedir = "data_17/v"+args.version+"/"
if os.path.isdir(basedir) is False:
    os.makedirs(basedir)

datasets = [
["EleMu_B",         "/MuonEG/Run2017B-" + campaign],
["EleMu_C",         "/MuonEG/Run2017C-" + campaign],
["EleMu_D",         "/MuonEG/Run2017D-" + campaign],
["EleMu_E",         "/MuonEG/Run2017E-" + campaign],
["EleMu_F",         "/MuonEG/Run2017F-" + campaign],
["DoubleMu_B",      "/DoubleMuon/Run2017B-" + campaign],
["DoubleMu_C",      "/DoubleMuon/Run2017C-" + campaign],
["DoubleMu_D",      "/DoubleMuon/Run2017D-" + campaign],
["DoubleMu_E",      "/DoubleMuon/Run2017E-" + campaign],
["DoubleMu_F",      "/DoubleMuon/Run2017F-" + campaign],
["DoubleEle_B",     "/DoubleEG/Run2017B-" + campaign],
["DoubleEle_C",     "/DoubleEG/Run2017C-" + campaign],
["DoubleEle_D",     "/DoubleEG/Run2017D-" + campaign],
["DoubleEle_E",     "/DoubleEG/Run2017E-" + campaign],
["DoubleEle_F",     "/DoubleEG/Run2017F-" + campaign],
["SingleMu_B",      "/SingleMuon/Run2017B-" + campaignSingleMu],
["SingleMu_C",      "/SingleMuon/Run2017C-" + campaignSingleMu],
["SingleMu_D",      "/SingleMuon/Run2017D-" + campaignSingleMu],
["SingleMu_E",      "/SingleMuon/Run2017E-" + campaignSingleMu],
["SingleMu_F",      "/SingleMuon/Run2017F-" + campaignSingleMu],
["SingleEle_B",     "/SingleElectron/Run2017B-" + campaign],
["SingleEle_C",     "/SingleElectron/Run2017C-" + campaign],
["SingleEle_D",     "/SingleElectron/Run2017D-" + campaign],
["SingleEle_E",     "/SingleElectron/Run2017E-" + campaign],
["SingleEle_F",     "/SingleElectron/Run2017F-" + campaign],
["MET_B",           "/MET/Run2017B-" + campaign],
["MET_C",           "/MET/Run2017C-" + campaign],
["MET_D",           "/MET/Run2017D-" + campaign],
["MET_E",           "/MET/Run2017E-" + campaign],
["MET_F",           "/MET/Run2017F-" + campaign],
]


for i in range(len(datasets)):
    file_name = basedir + datasets[i][0] + ".txt"
    for k in range(9):
        ds_name = datasets[i][1] + "-v" + str(k+1) + "/NANOAOD"
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
    if has_dataset:
        for dataset_name in reversed(k_lines):
            command = "dasgoclient --limit 0 --query 'file dataset=" + dataset_name + "' > " + file_name
            os.system(command)
            NumLines = sum(1 for line in open(file_name))
            if NumLines > 0:
                print(dataset_name)
                break
    else:
        open(file_name, 'a').close()
        print(datasets[i][1] + " is not available!")
    os.system("rm temp.txt")
