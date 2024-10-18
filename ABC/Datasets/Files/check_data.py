import os
import sys
import argparse

# List of datasets generated with the dasgoclient command.
# Example: dasgoclient --limit 0 --query 'dataset dataset=/MET*/*22*/NANOAOD'

#======SETUP=======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year")
args = parser.parse_args()

campaign = "*20"+args.year+"*/NANOAOD"

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
    command = "dasgoclient --limit 0 --query 'dataset dataset=/" + datasets[i] + "/*" + campaign + "'"
    print(">>>> " + command)
    os.system(command)
    print(" ")


