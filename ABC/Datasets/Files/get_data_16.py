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
    campaign = "UL2016_MiniAODv2_NanoAODv9"


basedir = "data_16/v"+args.version+"/"
if os.path.isdir(basedir) is False:
    os.makedirs(basedir)

datasets = [
["EleMu_B_v1",       "/MuonEG/Run2016B-ver1_HIPM_" + campaign],
["EleMu_B_v2",       "/MuonEG/Run2016B-ver2_HIPM_" + campaign],
["EleMu_C",          "/MuonEG/Run2016C-HIPM_" + campaign],
["EleMu_D",          "/MuonEG/Run2016D-HIPM_" + campaign],
["EleMu_E",          "/MuonEG/Run2016E-HIPM_" + campaign],
["EleMu_F_HIPM",     "/MuonEG/Run2016F-HIPM_" + campaign],
["EleMu_F",          "/MuonEG/Run2016F-" + campaign],
["EleMu_G",          "/MuonEG/Run2016G-" + campaign],
["EleMu_H",          "/MuonEG/Run2016H-" + campaign],
["DoubleMu_B_v1",    "/DoubleMuon/Run2016B-ver1_HIPM_" + campaign],
["DoubleMu_B_v2",    "/DoubleMuon/Run2016B-ver2_HIPM_" + campaign],
["DoubleMu_C",       "/DoubleMuon/Run2016C-HIPM_" + campaign],
["DoubleMu_D",       "/DoubleMuon/Run2016D-HIPM_" + campaign],
["DoubleMu_E",       "/DoubleMuon/Run2016E-HIPM_" + campaign],
["DoubleMu_F_HIPM",  "/DoubleMuon/Run2016F-HIPM_" + campaign],
["DoubleMu_F",       "/DoubleMuon/Run2016F-" + campaign],
["DoubleMu_G",       "/DoubleMuon/Run2016G-" + campaign],
["DoubleMu_H",       "/DoubleMuon/Run2016H-" + campaign],
["DoubleEle_B_v1",   "/DoubleEG/Run2016B-ver1_HIPM_" + campaign],
["DoubleEle_B_v2",   "/DoubleEG/Run2016B-ver2_HIPM_" + campaign],
["DoubleEle_C",      "/DoubleEG/Run2016C-HIPM_" + campaign],
["DoubleEle_D",      "/DoubleEG/Run2016D-HIPM_" + campaign],
["DoubleEle_E",      "/DoubleEG/Run2016E-HIPM_" + campaign],
["DoubleEle_F_HIPM", "/DoubleEG/Run2016F-HIPM_" + campaign],
["DoubleEle_F",      "/DoubleEG/Run2016F-" + campaign],
["DoubleEle_G",      "/DoubleEG/Run2016G-" + campaign],
["DoubleEle_H",      "/DoubleEG/Run2016H-" + campaign],
["SingleMu_B_v1",    "/SingleMuon/Run2016B-ver1_HIPM_" + campaign],
["SingleMu_B_v2",    "/SingleMuon/Run2016B-ver2_HIPM_" + campaign],
["SingleMu_C",       "/SingleMuon/Run2016C-HIPM_" + campaign],
["SingleMu_D",       "/SingleMuon/Run2016D-HIPM_" + campaign],
["SingleMu_E",       "/SingleMuon/Run2016E-HIPM_" + campaign],
["SingleMu_F_HIPM",  "/SingleMuon/Run2016F-HIPM_" + campaign],
["SingleMu_F",       "/SingleMuon/Run2016F-" + campaign],
["SingleMu_G",       "/SingleMuon/Run2016G-" + campaign],
["SingleMu_H",       "/SingleMuon/Run2016H-" + campaign],
["SingleEle_B_v1",   "/SingleElectron/Run2016B-ver1_HIPM_" + campaign],
["SingleEle_B_v2",   "/SingleElectron/Run2016B-ver2_HIPM_" + campaign],
["SingleEle_C",      "/SingleElectron/Run2016C-HIPM_" + campaign],
["SingleEle_D",      "/SingleElectron/Run2016D-HIPM_" + campaign],
["SingleEle_E",      "/SingleElectron/Run2016E-HIPM_" + campaign],
["SingleEle_F_HIPM", "/SingleElectron/Run2016F-HIPM_" + campaign],
["SingleEle_F",      "/SingleElectron/Run2016F-" + campaign],
["SingleEle_G",      "/SingleElectron/Run2016G-" + campaign],
["SingleEle_H",      "/SingleElectron/Run2016H-" + campaign],
["MET_B_v1",         "/MET/Run2016B-ver1_HIPM_" + campaign],
["MET_B_v2",         "/MET/Run2016B-ver2_HIPM_" + campaign],
["MET_C",            "/MET/Run2016C-HIPM_" + campaign],
["MET_D",            "/MET/Run2016D-HIPM_" + campaign],
["MET_E",            "/MET/Run2016E-HIPM_" + campaign],
["MET_F_HIPM",       "/MET/Run2016F-HIPM_" + campaign],
["MET_F",            "/MET/Run2016F-" + campaign],
["MET_G",            "/MET/Run2016G-" + campaign],
["MET_H",            "/MET/Run2016H-" + campaign],
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
