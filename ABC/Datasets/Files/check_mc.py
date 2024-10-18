import os
import sys
import argparse

# List of datasets generated with the dasgoclient command.
# Example: dasgoclient --limit 0 --query 'dataset dataset=/QCD*/*22*v12*/NANOAODSIM'

#======SETUP=======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version")
parser.set_defaults(version=None)
parser.add_argument("-y", "--year")
args = parser.parse_args()


if args.version is None:
    campaign = "*"+args.year+"*/NANOAODSIM"
else:
    args.version = "v"+args.version
    campaign = "*"+args.year+"*"+args.version+"*/NANOAODSIM"


datasets = [
"DYJetsToLL_LHEFilterPtZ-*_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
#Signal
"bbHToZaToLLChiChi_2HDMa"
]


print("#############################################################################################")
for i in range(len(datasets)):
    command = "dasgoclient --limit 0 --query 'dataset dataset=/" + datasets[i] + "*/" + campaign + "'"
    print(">>>> " + command)
    os.system(command)
    print(" ")
