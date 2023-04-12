import os
import sys
import argparse

# List of datasets generated with the command below (example for WWW). 
# dasgoclient --limit 0 --query 'dataset dataset=/WWW*/RunIISummer20UL18NanoAODv2-106X*/NANOAODSIM'

#======SETUP=======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version")
parser.add_argument("-p", "--period")
args = parser.parse_args()

if args.version != "8" and args.version != "9":
    print("Please, enter a valid version: 8 or 9")
    sys.exit()
    
if args.period != "16" and args.period != "17" and args.period != "18":
    print("Please, enter a valid period: 16, 17, or 18") 
    sys.exit()

if args.version == "8":
    args.version = "v2"
elif args.version == "9":
    args.version = "v9"

campaign = "Run20"+args.period+"*NanoAOD"+args.version+"*/NANOAOD"

datasets = [
"MuonEG",					
"DoubleMuon",
"DoubleEG",
"SingleMuon",
"SingleElectron",
"MET",
"EGamma",
]

print("#############################################################################################")
for i in range(len(datasets)):
    print(">>>> " + datasets[i])
    command = "dasgoclient --limit 0 --query 'dataset dataset=/" + datasets[i] + "/*" + campaign + "'"
    os.system(command)
    print(" ")



# dasgoclient --limit 0 --query 'dataset dataset=/MuonEG/*UL2016_MiniAODv1_NanoAODv2*/NANOAOD'
