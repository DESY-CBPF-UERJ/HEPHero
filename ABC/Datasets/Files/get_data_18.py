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
    campaign = "UL2018_MiniAODv2_NanoAODv9_GT36"
    campaignEleD = "UL2018_MiniAODv2_NanoAODv9"


basedir = "data_18/v"+args.version+"/"
if os.path.isdir(basedir) is False:
    os.makedirs(basedir)

datasets = [
["EleMu_A",         "/MuonEG/Run2018A-" + campaign],   
["EleMu_B",         "/MuonEG/Run2018B-" + campaign],
["EleMu_C",         "/MuonEG/Run2018C-" + campaign],
["EleMu_D",         "/MuonEG/Run2018D-" + campaign],
["DoubleMu_A",      "/DoubleMuon/Run2018A-" + campaign],
["DoubleMu_B",      "/DoubleMuon/Run2018B-" + campaign],
["DoubleMu_C",      "/DoubleMuon/Run2018C-" + campaign],
["DoubleMu_D",      "/DoubleMuon/Run2018D-" + campaign],
["SingleMu_A",      "/SingleMuon/Run2018A-" + campaign],
["SingleMu_B",      "/SingleMuon/Run2018B-" + campaign],
["SingleMu_C",      "/SingleMuon/Run2018C-" + campaign],
["SingleMu_D",      "/SingleMuon/Run2018D-" + campaign],
["Ele_A",           "/EGamma/Run2018A-" + campaign],
["Ele_B",           "/EGamma/Run2018B-" + campaign],
["Ele_C",           "/EGamma/Run2018C-" + campaign],
["Ele_D",           "/EGamma/Run2018D-" + campaignEleD],
["MET_A",           "/MET/Run2018A-" + campaign],
["MET_B",           "/MET/Run2018B-" + campaign],
["MET_C",           "/MET/Run2018C-" + campaign],
["MET_D",           "/MET/Run2018D-" + campaign],
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
