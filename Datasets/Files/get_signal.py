import os
import sys
import argparse

# List of datasets generated with the command below (example for WWW). 
# dasgoclient --limit 0 --query 'dataset dataset=/WWW*/RunIISummer20UL16*NanoAODv2-106X*/NANOAODSIM'

#======SETUP=======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version")
parser.add_argument("-p", "--period")
parser.add_argument("--apv", dest='apv', action='store_true')
parser.set_defaults(apv=False)
args = parser.parse_args()


if args.version != "8" and args.version != "9":
    print("Please, enter a valid version: 8 or 9")
    sys.exit()
    
if args.period != "16" and args.period != "17" and args.period != "18":
    print("Please, enter a valid period: 16, 17, or 18") 
    sys.exit()


if args.period == "16":
    if args.apv:
        if args.version == "8":
            version = "APVv2"
            real_version = "APVv8"
            tag = "mcRun2_asymptotic_preVFP_v9"
        elif args.version == "9":
            version = "APVv9"
            real_version = "APVv9"
            tag = "mcRun2_asymptotic_preVFP_v11"
    else:
        if args.version == "8":
            version = "v2"
            real_version = "v8"
            tag = "mcRun2_asymptotic_v15"
        elif args.version == "9":
            version = "v9"
            real_version = "v9"
            tag = "mcRun2_asymptotic_v17"

elif args.period == "17":
    if args.version == "8":
        version = "v2"
        real_version = "v8"
        tag = "mc2017_realistic_v8"
    elif args.version == "9":
        version = "v9"
        real_version = "v9"
        tag = "mc2017_realistic_v9"

elif args.period == "18":
    if args.version == "8":
        version = "v2"
        real_version = "v8"
        tag = "upgrade2018_realistic_v15_L1v1"
    elif args.version == "9":
        version = "v9"
        real_version = "v9"
        tag = "upgrade2018_realistic_v16_L1v1"        
  

campaign = "RunIISummer20UL"+args.period+"NanoAOD"+version+"-106X_"+tag
#campaign_ext = "RunIISummer20UL"+args.period+"NanoAOD"+version+"-106X_"+tag+"_ext1"

basedir = "signal_"+args.period+"/UL_"+real_version+"/"
if os.path.isdir(basedir) is False:
    os.makedirs(basedir)
    
datasets = [
["Signal_400_100",          ["/bbHToZaToLLChiChi_2HDMa_MH-400_Ma-100_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_400_200",          ["/bbHToZaToLLChiChi_2HDMa_MH-400_Ma-200_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_500_100",          ["/bbHToZaToLLChiChi_2HDMa_MH-500_Ma-100_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_500_200",          ["/bbHToZaToLLChiChi_2HDMa_MH-500_Ma-200_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_500_300",          ["/bbHToZaToLLChiChi_2HDMa_MH-500_Ma-300_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_600_100",          ["/bbHToZaToLLChiChi_2HDMa_MH-600_Ma-100_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_600_200",          ["/bbHToZaToLLChiChi_2HDMa_MH-600_Ma-200_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_600_300",          ["/bbHToZaToLLChiChi_2HDMa_MH-600_Ma-300_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_600_400",          ["/bbHToZaToLLChiChi_2HDMa_MH-600_Ma-400_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_800_100",          ["/bbHToZaToLLChiChi_2HDMa_MH-800_Ma-100_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_800_200",          ["/bbHToZaToLLChiChi_2HDMa_MH-800_Ma-200_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_800_300",          ["/bbHToZaToLLChiChi_2HDMa_MH-800_Ma-300_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_800_400",          ["/bbHToZaToLLChiChi_2HDMa_MH-800_Ma-400_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_800_600",          ["/bbHToZaToLLChiChi_2HDMa_MH-800_Ma-600_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_1000_100",         ["/bbHToZaToLLChiChi_2HDMa_MH-1000_Ma-100_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_1000_200",         ["/bbHToZaToLLChiChi_2HDMa_MH-1000_Ma-200_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_1000_300",         ["/bbHToZaToLLChiChi_2HDMa_MH-1000_Ma-300_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_1000_400",         ["/bbHToZaToLLChiChi_2HDMa_MH-1000_Ma-400_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_1000_600",         ["/bbHToZaToLLChiChi_2HDMa_MH-1000_Ma-600_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
["Signal_1000_800",         ["/bbHToZaToLLChiChi_2HDMa_MH-1000_Ma-800_MChi-45_TuneCP5_13TeV_madgraph-pythia8/" + campaign]],
]


for i in range(len(datasets)):
    file_name = basedir + datasets[i][0] + ".txt"
    for j in range(len(datasets[i][1])):
        for k in range(9):
            ds_name = datasets[i][1][j] + "-v" + str(k+1) + "/NANOAODSIM"
            if k == 0:
                command = "dasgoclient --limit 0 --query 'dataset dataset=" + ds_name + "' > " + "temp.txt"
            else:
                command = "dasgoclient --limit 0 --query 'dataset dataset=" + ds_name + "' >> " + "temp.txt"
            os.system(command)
        has_dataset = False
        k_lines = []
        with open("temp.txt", "r") as file:
            for line in file:
                pass
                has_dataset = True
                k_lines.append(line[0:-1])
        if has_dataset:
            for dataset_name in reversed(k_lines):
                if j == 0:
                    command = "dasgoclient --limit 0 --query 'file dataset=" + dataset_name + "' > " + file_name
                else:
                    command = "dasgoclient --limit 0 --query 'file dataset=" + dataset_name + "' >> " + file_name
                os.system(command)
                NumLines = sum(1 for line in open(file_name))
                if NumLines > 0:
                    print(dataset_name)
                    break
        else:
            open(file_name, 'a').close()
            print(datasets[i][1][j] + " is not available!")
        os.system("rm temp.txt")
            
