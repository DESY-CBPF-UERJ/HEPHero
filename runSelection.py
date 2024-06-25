#!/bin/bash
import os
import time
import sys
import argparse
import warnings
import json
from shutil import copyfile
import glob
import glob
import glob
import ast
import h5py
import pandas as pd
import numpy as np

#-------------------------------------------------------------------------------------
# Main Setup
#-------------------------------------------------------------------------------------
selection = 'Test'
analysis = 'HHDM'
treeName = 'Events'
LumiWeights = 1

analysis_parameters = {
'JET_ETA_CUT                ': 2.4,
'JET_PT_CUT                 ': 20,
'JET_ID_WP                  ': 6,
'JET_PUID_WP                ': 7,
'JET_BTAG_WP                ': 3,
'JET_LEP_DR_ISO_CUT         ': 0.4,
'ELECTRON_GAP_LOWER_CUT     ': 1.444,
'ELECTRON_GAP_UPPER_CUT     ': 1.566,
'ELECTRON_ETA_CUT           ': 2.5,
'ELECTRON_PT_CUT            ': 20,
'ELECTRON_LOW_PT_CUT        ': 15,
'ELECTRON_ID_WP             ': 4,
'MUON_ETA_CUT               ': 2.4,
'MUON_PT_CUT                ': 20,
'MUON_LOW_PT_CUT            ': 15,
'MUON_ID_WP                 ': 1,
'MUON_ISO_WP                ': 3,
'LEADING_LEP_PT_CUT         ': 40,
'LEPLEP_PT_CUT              ': 40,
'LEPLEP_DR_CUT              ': 3.2,
'LEPLEP_DM_CUT              ': 25,
'MET_CUT                    ': 65,
'MET_DY_UPPER_CUT           ': 100,
'MET_LEPLEP_DPHI_CUT        ': 0.8,
'MET_LEPLEP_MT_CUT          ': 90,
}

corrections = {  # 0-don't apply, 1-apply
'PILEUP_WGT                 ': 1,
'ELECTRON_ID_WGT            ': 1,
'MUON_ID_WGT                ': 1,
'JET_PUID_WGT               ': 1,
'BTAG_WGT                   ': 1,
'TRIGGER_WGT                ': 1,
'PREFIRING_WGT              ': 1,
'JER_CORR                   ': 1,
'MET_XY_CORR                ': 1,
'MET_RECOIL_CORR            ': 1,
'TOP_PT_WGT                 ': 1,
'W_PT_WGT                   ': 1,
'MUON_ROC_CORR              ': 1,
}


#-------------------------------------------------------------------------------------
# Systematics
#-------------------------------------------------------------------------------------
lateral_systematics = {
'CV': [0, 1, [], []],
'JER': [28, 2, [], []],
'UncMET': [29, 2, [], []],
'Recoil': [30, 6, ['02'], []],
'JES': [41, 2, [], ['Total']],
}

vertical_systematics = {
'Pileup': [50, 2, [], []],
'EleID': [51, 2, [], []],
'MuID': [52, 2, [], []],
'JetPU': [53, 2, [], []],
'BTag': [54, 8, [], ['bc', 'light', 'bc_fc', 'light_fc']],
'Trig': [58, 2, [], []],
'PreFir': [59, 2, [], []],
'PDF': [60, 100, [], []],
'Scales': [62, 7, [], []],
'ISR': [63, 2, [], []],
'FSR': [64, 2, [], []],
'TopPt': [65, 1, [], []],
'WPt': [66, 1, [], []],
}


#-------------------------------------------------------------------------------------
# Jobs setup
#-------------------------------------------------------------------------------------
NumMaxEvents = -1
NumFilesPerJob_Data = 2
NumFilesPerJob_Signal = 50
NumFilesPerJob_Bkg = 4


#-------------------------------------------------------------------------------------
# Datasets
#-------------------------------------------------------------------------------------
sys.path.insert(0, 'Datasets')
from Signal import *
from Bkg import *
from Data import *
datasets = []

#datasets.extend(Data_Lep_preVFP_16)
#datasets.extend(Data_MET_preVFP_16)
#datasets.extend(Signal_preVFP_16)
#datasets.extend(DYPt50ToInf_preVFP_16)
#datasets.extend(DYPt0To50_preVFP_16)
#datasets.extend(TTFullLep_preVFP_16)
#datasets.extend(TTSemiLep_preVFP_16)
#datasets.extend(ST_preVFP_16)
#datasets.extend(VZ_preVFP_16)
#datasets.extend(ResidualSM_preVFP_16)

#datasets.extend(Data_Lep_postVFP_16)
#datasets.extend(Data_MET_postVFP_16)
#datasets.extend(Signal_postVFP_16)
#datasets.extend(DYPt50ToInf_postVFP_16)
#datasets.extend(DYPt0To50_postVFP_16)
#datasets.extend(TTFullLep_postVFP_16)
#datasets.extend(TTSemiLep_postVFP_16)
#datasets.extend(ST_postVFP_16)
#datasets.extend(VZ_postVFP_16)
#datasets.extend(ResidualSM_postVFP_16)

#datasets.extend(Data_Lep_17)
#datasets.extend(Data_MET_17)
#datasets.extend(Signal_17)
#datasets.extend(DYPt50ToInf_17)
#datasets.extend(DYPt0To50_17)
#datasets.extend(TTFullLep_17)
#datasets.extend(TTSemiLep_17)
#datasets.extend(ST_17)
#datasets.extend(VZ_17)
#datasets.extend(ResidualSM_17)

datasets.extend(Data_Lep_18)
#datasets.extend(Data_MET_18)
datasets.extend(Signal_18)
datasets.extend(DYPt50ToInf_18)
datasets.extend(DYPt0To50_18)
datasets.extend(TTFullLep_18)
datasets.extend(TTSemiLep_18)
datasets.extend(ST_18)
datasets.extend(VZ_18)
datasets.extend(ResidualSM_18)


#-------------------------------------------------------------------------------------
# Metadata
#-------------------------------------------------------------------------------------
metadata = {
'NN_prep_keras_XX         ': 'Metadata/ML/Keras/preprocessing.json',
'NN_model_keras_XX        ': 'Metadata/ML/Keras/NN_4_100_elu_adam/model.json',
'NN_model_torch_HIPM_16   ': 'Metadata/ML/Torch/DeepCSV/2016preVFP/model_scripted.pt',
'NN_model_torch_NORMAL_16 ': 'Metadata/ML/Torch/DeepCSV/2016postVFP/model_scripted.pt',
'NN_model_torch_17        ': 'Metadata/ML/Torch/DeepCSV/2017/model_scripted.pt',
'NN_model_torch_18        ': 'Metadata/ML/Torch/DeepCSV/2018/model_scripted.pt',
'lumi_certificate_16      ': 'Metadata/certificates/Cert_271036-284044_13TeV_Legacy2016_Collisions16.json',
'lumi_certificate_17      ': 'Metadata/certificates/Cert_294927-306462_13TeV_UL2017_Collisions17.json',
'lumi_certificate_18      ': 'Metadata/certificates/Cert_314472-325175_13TeV_Legacy2018_Collisions18.json',
'pdf_type_XX              ': 'Metadata/PDF/pdf_type.json',
'pileup_HIPM_16           ': 'Metadata/POG/LUM/2016preVFP_UL/puWeights.json.gz',
'pileup_NORMAL_16         ': 'Metadata/POG/LUM/2016postVFP_UL/puWeights.json.gz',
'pileup_17                ': 'Metadata/POG/LUM/2017_UL/puWeights.json.gz',
'pileup_18                ': 'Metadata/POG/LUM/2018_UL/puWeights.json.gz',
'electron_HIPM_16         ': 'Metadata/POG/EGM/2016preVFP_UL/electron.json.gz',
'electron_NORMAL_16       ': 'Metadata/POG/EGM/2016postVFP_UL/electron.json.gz',
'electron_17              ': 'Metadata/POG/EGM/2017_UL/electron.json.gz',
'electron_18              ': 'Metadata/POG/EGM/2018_UL/electron.json.gz',
'muon_HIPM_16             ': 'Metadata/POG/MUO/2016preVFP_UL/muon_Z.json.gz',
'muon_NORMAL_16           ': 'Metadata/POG/MUO/2016postVFP_UL/muon_Z.json.gz',
'muon_17                  ': 'Metadata/POG/MUO/2017_UL/muon_Z.json.gz',
'muon_18                  ': 'Metadata/POG/MUO/2018_UL/muon_Z.json.gz',
'btag_SF_HIPM_16          ': 'Metadata/POG/BTV/2016preVFP_UL/btagging.json.gz',
'btag_SF_NORMAL_16        ': 'Metadata/POG/BTV/2016postVFP_UL/btagging.json.gz',
'btag_SF_17               ': 'Metadata/POG/BTV/2017_UL/btagging.json.gz',
'btag_SF_18               ': 'Metadata/POG/BTV/2018_UL/btagging.json.gz',
'btag_eff_HIPM_16         ': 'Metadata/btag_eff/DeepCSVLoose/2016preVFP.json',
'btag_eff_NORMAL_16       ': 'Metadata/btag_eff/DeepCSVLoose/2016postVFP.json',
'btag_eff_17              ': 'Metadata/btag_eff/DeepCSVLoose/2017.json',
'btag_eff_18              ': 'Metadata/btag_eff/DeepCSVLoose/2018.json',
'trigger_16               ': 'Metadata/trigger/SF_2016_ttbar.json',
'trigger_17               ': 'Metadata/trigger/SF_2017_ttbar.json',
'trigger_18               ': 'Metadata/trigger/SF_2018_ttbar.json',
'JES_MC_HIPM_16           ': 'Metadata/JES/JES_MC_16_preVFP.txt',
'JES_MC_NORMAL_16         ': 'Metadata/JES/JES_MC_16_postVFP.txt',
'JES_MC_17                ': 'Metadata/JES/JES_MC_17.txt',
'JES_MC_18                ': 'Metadata/JES/JES_MC_18.txt',
'JER_MC_HIPM_16           ': 'Metadata/JER/Summer20UL16APV_JRV3_MC/Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs.txt',
'JER_SF_MC_HIPM_16        ': 'Metadata/JER/Summer20UL16APV_JRV3_MC/Summer20UL16APV_JRV3_MC_SF_AK4PFchs.txt',
'JER_MC_NORMAL_16         ': 'Metadata/JER/Summer20UL16_JRV3_MC/Summer20UL16_JRV3_MC_PtResolution_AK4PFchs.txt',
'JER_SF_MC_NORMAL_16      ': 'Metadata/JER/Summer20UL16_JRV3_MC/Summer20UL16_JRV3_MC_SF_AK4PFchs.txt',
'JER_MC_17                ': 'Metadata/JER/Summer19UL17_JRV2_MC/Summer19UL17_JRV2_MC_PtResolution_AK4PFchs.txt',
'JER_SF_MC_17             ': 'Metadata/JER/Summer19UL17_JRV2_MC/Summer19UL17_JRV2_MC_SF_AK4PFchs.txt',
'JER_MC_18                ': 'Metadata/JER/Summer19UL18_JRV2_MC/Summer19UL18_JRV2_MC_PtResolution_AK4PFchs.txt',
'JER_SF_MC_18             ': 'Metadata/JER/Summer19UL18_JRV2_MC/Summer19UL18_JRV2_MC_SF_AK4PFchs.txt',
'jet_puID_HIPM_16         ': 'Metadata/POG/JME/2016preVFP_UL/jmar.json.gz',
'jet_puID_NORMAL_16       ': 'Metadata/POG/JME/2016postVFP_UL/jmar.json.gz',
'jet_puID_17              ': 'Metadata/POG/JME/2017_UL/jmar.json.gz',
'jet_puID_18              ': 'Metadata/POG/JME/2018_UL/jmar.json.gz',
'JERC_HIPM_16             ': 'Metadata/POG/JME/2016preVFP_UL/jet_jerc.json.gz',
'JERC_NORMAL_16           ': 'Metadata/POG/JME/2016postVFP_UL/jet_jerc.json.gz',
'JERC_17                  ': 'Metadata/POG/JME/2017_UL/jet_jerc.json.gz',
'JERC_18                  ': 'Metadata/POG/JME/2018_UL/jet_jerc.json.gz',
'mu_RoccoR_HIPM_16        ': 'Metadata/mu_Rochester/RoccoR2016aUL.txt',
'mu_RoccoR_NORMAL_16      ': 'Metadata/mu_Rochester/RoccoR2016bUL.txt',
'mu_RoccoR_17             ': 'Metadata/mu_Rochester/RoccoR2017UL.txt',
'mu_RoccoR_18             ': 'Metadata/mu_Rochester/RoccoR2018UL.txt',
'Z_recoil_16              ': 'Metadata/boson_recoil/Z/TypeI-PFMet_Run2016_legacy.root',
'Z_recoil_17              ': 'Metadata/boson_recoil/Z/Type1_PFMET_2017.root',
'Z_recoil_18              ': 'Metadata/boson_recoil/Z/TypeI-PFMet_Run2018.root',
}


#-------------------------------------------------------------------------------------
# Plots
#-------------------------------------------------------------------------------------
Get_Image_in_EPS = 0
Get_Image_in_PNG = 1
Get_Image_in_PDF = 0


#-------------------------------------------------------------------------------------
# [DO NOT TOUCH THIS PART] 
#------------------------------------------------------------------------------------------------------------

#======ARGUMENTS SETUP=============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--job", type=int, default=0)
parser.add_argument("-p", "--proxy", default="none")
parser.add_argument("-t", "--timer", type=int, default=1)
parser.add_argument("-c", "--check", type=int, default=-1)
parser.add_argument("--resubmit", dest='resubmit_flag', action='store_true')
parser.set_defaults(resubmit_flag=False)
parser.add_argument("--fix", dest='fix_flag', action='store_true')
parser.set_defaults(fix_flag=False)
parser.add_argument("--start", dest='start_flag', action='store_true')
parser.set_defaults(start_flag=False)

args = parser.parse_args()
print('Job ID = ' + str(args.job))
print('Timer = ' + str(args.timer))
print('Proxy = ' + args.proxy)   
print('Check = ' + str(args.check))
print('')
if args.proxy != "none":
    os.environ["X509_USER_PROXY"] = args.proxy


#======RESUBMIT SETUP==============================================================================
resubmit = []
if args.resubmit_flag:
    resubmit_files = glob.glob("tools/resubmit_*")
    for rfile in resubmit_files:
        with open(rfile) as f:
            for line in f:
                line = ast.literal_eval(line[:-1])[0]
                resubmit.append(line)


#======SETUP ENVIRONMENT VARIABLES=================================================================
user = os.environ.get("USER")
machines = os.environ.get("MACHINES")

  
#======CASE SELECTION==============================================================================
if args.check >= 0:
    outpath = "ana/local_output"
    redirector = "None"
    N = 0
    
    if analysis == "GEN":
        if args.check == 0:
            datasets = [["H7_test", '1600000', "Datasets/test.hepmc", 0],]
            #datasets = [["H7_cms", '1600000', "Datasets/InterfaceMatchboxTest-S123456790_cms.hepmc", 0],]
            #datasets = [["H7_dell", '1600000', "Datasets/InterfaceMatchboxTest-S123456790_dell.hepmc", 0],]
            #datasets = [["H7_ttbar_tests", '1600000', "/home/gcorreia/cernbox/ttbar_test/LHC-Matchbox.hepmc", 0],]
        if args.check >= 1:
            sys.exit("There are only 1 test from 0 to 0.")
    else:
        #if args.check == 0:
        #    datasets = [["Signal_test_16",                  '1600000', "../splitSUSY_M1400_1300_ctau0p1.root", 0],]
        #if args.check == 1:
        #    datasets = [["TTToSemiLeptonic_test_17",        '1700000', "../TTToSemiLeptonic.root",  0],]
        #if args.check == 2:
        #    datasets = [["TTToSemiLeptonic_test_18",        '1800000', "../ZJetsToNuNu.root",  0],]
        if args.check == 0:
            datasets = [["Signal_test_16",                  '1600000', "Datasets/signal_nano.root", 0],]
        if args.check == 1:
            datasets = [["TTToSemiLeptonic_test_APV_16",    '1600001', "Datasets/ttbar_nano.root",  0],]
        if args.check == 2:
            datasets = [["TTToSemiLeptonic_test_16",        '1600000', "Datasets/ttbar_nano.root",  0],]
        if args.check == 3:
            datasets = [["TTToSemiLeptonic_test_17",        '1700000', "Datasets/ttbar_nano.root",  0],]
        if args.check == 4:
            datasets = [["TTToSemiLeptonic_test_18",        '1800000', "Datasets/ttbar_nano.root",  0],]
        if args.check == 5:
            datasets = [["Data_DoubleMu_test_H_16",         '1600000', "Datasets/data_nano.root",   0],]
        if args.check >= 6:
            sys.exit("There are only 6 tests from 0 to 5.")
    
    jobs = [[datasets[0], 0, 1, 0, 0]]
    
    #with open('tools/analysis.txt', "w") as newfile:
    #    newfile.write(analysis)
else:
    #======SETUP ENVIRONMENT VARIABLES======
    hep_outpath = os.environ.get("HEP_OUTPATH")
    redirector = os.environ.get("REDIRECTOR")

    
    if user is None:
        raise ValueError("USER environment variable is undefined. Aborting script execution...")
    first_letter_user = user[0]

    if machines is None:
        raise ValueError("MACHINES environment variable is undefined. Aborting script execution...")

    if hep_outpath is None:
        hep_outpath = os.path.join("/eos/user", first_letter_user, user, "HEP_OUTPUT")
        #hep_outpath = "/home/" # used for offline tests
        warnings.warn("HEP_OUTPATH environment varibale is undefined. Defaulting to " + hep_outpath)
        if os.path.isdir(hep_outpath) is False:
            try:
                os.makedirs(hep_outpath)
            except Exception as e:
                quit("Script failed when creating default output path. Exception: " + str(e) + " Aborting...")
    
    outpath = os.path.join(hep_outpath, analysis)
    if os.path.isdir(outpath) is False:
        try:
            os.makedirs(outpath)
        except Exception as e:
            quit("Script failed when creating default output path. Exception: " + str(e) + " Aborting...")

    if redirector is None:
        # Possible redirectors: ['cmsxrootd.fnal.gov', 'xrootd-cms.infn.it', 'cms-xrd-global.cern.ch']
        # recommended in [USA, Europe/Asia, Global]
        redirector = 'cms-xrd-global.cern.ch'
        warnings.warn("REDIRECTOR environment varibale is undefined. Defaulting to " + redirector)
    
 
    print("\nuser = " + user)
    print("machines = " + machines)
    print("outpath = " + outpath)
    print("redirector = " + redirector)
    
    #======CREATE LIST OF JOBS======
    if analysis == "GEN":
        jobs = []
        empty_text_files = []
        total_number_of_files = {}
        for dataset in datasets:
            NumFilesPerJob = 1
            
            NumFiles = sum(1 for line in open(dataset[2]))
            total_number_of_files[dataset[0]] = NumFiles
            
            if NumFiles == 0:
                empty_text_files.append(dataset[0])
            else:
                dataset.append(NumFiles)

                Intervals = list(range(0,NumFiles,NumFilesPerJob))
                if NumFiles%NumFilesPerJob == 0:
                    Intervals.append(Intervals[-1]+NumFilesPerJob)
                else:
                    Intervals.append(Intervals[-1]+NumFiles%NumFilesPerJob)

                for i in range(len(Intervals)-1):
                    jobs += [ [dataset, Intervals[i], Intervals[i+1], 0, 0] ]
                

        if len(resubmit) > 0:
            jobs = resubmit
        if args.resubmit_flag and len(resubmit) == 0:
            print("Warning: there is no job to be resubmitted, it is being considered the initial list of jobs.")
        
        N = int(args.job)
        if N == -1:
            print("")
            sys.exit("Number of jobs: " + str(len(jobs)))
        if N == -2:
            for i in range(len(jobs)):
                print(str(i)+"  "+str(jobs[i])+",")
            sys.exit("")
        else:
            if N <= -3:
                print("")
                sys.exit(">> Enter an integer >= -2")
        if N >= len(jobs):
            sys.exit("There are only " + str(len(jobs)) + " jobs")
    else:
        jobs = []
        empty_text_files = []
        files_not_at_local_storage = []
        number_of_files_not_at_local_storage = {}
        total_number_of_files = {}
        for dataset in datasets:
            files_not_at_local_storage_per_dataset = []
            if( dataset[0][:4] == "Data" ):
                NumFilesPerJob = NumFilesPerJob_Data;
            elif( dataset[0][:6] == "Signal" ):
                NumFilesPerJob = NumFilesPerJob_Signal;
            else:
                NumFilesPerJob = NumFilesPerJob_Bkg;
            
            NumFiles = sum(1 for line in open(dataset[2]))
            total_number_of_files[dataset[0]] = NumFiles
            if NumFiles == 0:
                empty_text_files.append(dataset[0])
                number_of_files_not_at_local_storage[dataset[0]] = 0
            else:
                if machines != "CERN":
                    file_input = open(dataset[2], 'r')
                    lines = file_input.readlines()
                    lines = [x.strip() for x in lines]
                    NumFiles = 0
                    for line in lines:
                        if machines == "DESY":
                            file_path = "/pnfs/desy.de/cms/tier2/" + line
                        if machines == "UERJ":
                            file_path = "/mnt/hadoop/cms/" + line
                        if analysis == "OPENDATA":
                            file_path = hep_outpath[:-7] + "/opendata/" + line

                        if os.path.isfile(file_path) or dataset[1][2:4] == "99":
                            NumFiles += 1
                        else:
                            files_not_at_local_storage.append(line)
                            files_not_at_local_storage_per_dataset.append(line)
                    number_of_files_not_at_local_storage[dataset[0]] = len(files_not_at_local_storage_per_dataset)
                
                if NumFiles > 0:
                    dataset.append(NumFiles)

                    Intervals = list(range(0,NumFiles,NumFilesPerJob))
                    if NumFiles%NumFilesPerJob == 0:
                        Intervals.append(Intervals[-1]+NumFilesPerJob)
                    else:
                        Intervals.append(Intervals[-1]+NumFiles%NumFilesPerJob)
        
                    for i in range(len(Intervals)-1):
                        if( dataset[0][:4] == "Data" ):
                            jobs += [ [dataset, Intervals[i], Intervals[i+1], 0, 0] ]
                        else:
                            for systematic in lateral_systematics.keys():
                                if( len(lateral_systematics[systematic][2]) == 0 ):
                                    jobs += [ [dataset, Intervals[i], Intervals[i+1], lateral_systematics[systematic][0], u] for u in range(lateral_systematics[systematic][1]) ]
                                else:
                                    if( dataset[1][2:4] in lateral_systematics[systematic][2] ):
                                        jobs += [ [dataset, Intervals[i], Intervals[i+1], lateral_systematics[systematic][0], u] for u in range(lateral_systematics[systematic][1]) ]
                

        if len(resubmit) > 0:
            jobs = resubmit
        if args.resubmit_flag and len(resubmit) == 0:
            print("Warning: there is no job to be resubmitted, it is being considered the initial list of jobs.")
        
        N = int(args.job)
        if N == -1:
            print("")
            sys.exit("Number of jobs: " + str(len(jobs)))
        if N == -2:
            for i in range(len(jobs)):
                print(str(i)+"  "+str(jobs[i])+",")
            sys.exit("")
        if N == -3:
            print("")
            for i in range(len(empty_text_files)):
                print(empty_text_files[i])
            print("")
            sys.exit("There are " + str(len(empty_text_files)) + " empty text files")
        if machines != "CERN":
            if N == -4:
                for i in range(len(files_not_at_local_storage)):
                    print(files_not_at_local_storage[i])
                print("")    
                for dataset in datasets:
                    if number_of_files_not_at_local_storage[dataset[0]] > 0:
                        print(dataset[0] + " has " + str(number_of_files_not_at_local_storage[dataset[0]]) + " missing files of " + str(total_number_of_files[dataset[0]]))
                print("")
                sys.exit("There are " + str(len(files_not_at_local_storage)) + " missing files at local storage")
            if N <= -5:
                print("")
                sys.exit(">> Enter an integer >= -4")
        else:
            if N <= -4:
                print("")
                sys.exit(">> Enter an integer >= -3")
        if N >= len(jobs):
            sys.exit("There are only " + str(len(jobs)) + " jobs")
            

#======CREATE OUTPUT DIRECTORY FOR THE SELECTION AND COPY FILES THERE==============================
if not os.path.exists(os.path.join(outpath, selection)):
    os.makedirs(os.path.join(outpath, selection))
    
if args.fix_flag or args.start_flag:
    copyfile("ana/"+selection+".cpp", outpath+'/'+selection+"/"+selection+".cpp")
    
    for ijob in range(len(jobs)):
        job_dir = os.path.join(outpath, selection, jobs[ijob][0][0] + "_files_" + str(jobs[ijob][1]) + "_" + str(jobs[ijob][2]-1))
        if not os.path.exists(job_dir):    
            os.makedirs(job_dir)  
    
    jobs_file = open(outpath+'/'+selection+"/"+"jobs.txt", "w")
    for i in range(len(jobs)):
        #jobs_file.write(jobs[i][0][0] + "_files_" + str(jobs[i][1]) + "_" + str(jobs[i][2]-1)+"\n")
        jobs_file.write(str(i)+"  "+str(jobs[i])+","+"\n")
    jobs_file.close()
    
    json_sys_file = outpath+'/'+selection+"/"+'vertical_systematics.json'
    with open(json_sys_file, 'w') as fvs:
        json.dump(vertical_systematics, fvs)
    json_sys_file = outpath+'/'+selection+"/"+'lateral_systematics.json'    
    with open(json_sys_file, 'w') as fls:
        json.dump(lateral_systematics, fls)
      
if args.fix_flag:
    sys.exit()
    
if args.start_flag:
    for ijob in range(len(jobs)):
        job_dir = os.path.join(outpath, selection, jobs[ijob][0][0] + "_files_" + str(jobs[ijob][1]) + "_" + str(jobs[ijob][2]-1))
        job_sysID = jobs[ijob][3]
        job_universe = jobs[ijob][4]
        if job_sysID == 0:
            os.system("rm -rf " + job_dir + "/cutflow.txt")
            os.system("rm -rf " + job_dir + "/Histograms.root")
            os.system("rm -rf " + job_dir + "/selection.h5")
            os.system("rm -rf " + job_dir + "/Tree.root")
            os.system("rm -rf " + job_dir + "/Systematics/0_0.*")
        else:
            os.system("rm -rf " + job_dir + "/Systematics/" + str(job_sysID) + "_" + str(job_universe) + ".*")
    sys.exit()

output_dir = os.path.join(outpath, selection, jobs[N][0][0] + "_files_" + str(jobs[N][1]) + "_" + str(jobs[N][2]-1))
if not os.path.exists(output_dir):    
    os.makedirs(output_dir)
    
    
#======WRITE INPUT FILE OF THE SELECTION===========================================================
ConfigFile = "Metadata/ConfigFile_" + jobs[N][0][0] + "_" + str(jobs[N][1]) + "_" + str(jobs[N][2]-1) + "_" + str(jobs[N][3]) + "_" + str(jobs[N][4]) + ".txt"
in_file = open(ConfigFile, "w")
in_file.write("Selection            " + selection                              + "\n")
in_file.write("Analysis             " + analysis                               + "\n")
in_file.write("Outpath              " + outpath                                + "\n")
in_file.write("InputTree            " + treeName                               + "\n")
in_file.write("DatasetName          " + jobs[N][0][0]                          + "\n")
in_file.write("Files                " + str(jobs[N][1])+"_"+str(jobs[N][2]-1)  + "\n")
in_file.write("DatasetID            " + jobs[N][0][1]                          + "\n")
in_file.write("Redirector           " + str(redirector)                        + "\n")
in_file.write("Machines             " + str(machines)                          + "\n")
    
if args.check >= 0:
    in_file.write("Check                " + str(1)                             + "\n")
    in_file.write("InputFile            " + jobs[N][0][2]                      + "\n")
else:
    in_file.write("Check                " + str(0)                             + "\n")
    file_input = open(jobs[N][0][2], 'r')
    lines = file_input.readlines()
    lines = [x.strip() for x in lines]
    iline = 0
    for line in lines:
        if machines != "CERN" and analysis != "GEN":
            if line in files_not_at_local_storage:
                continue
        if iline >= jobs[N][1] and iline < jobs[N][2]:
            in_file.write("InputFile            " + line                       + "\n")
        iline += 1

if( jobs[N][0][0][:4] == "Data" ):
    in_file.write("NumMaxEvents         " + str(-1)                            + "\n")
    in_file.write("LumiWeights          " + str(0)                             + "\n")
    in_file.write("Universe             " + str(0)                             + "\n")
    in_file.write("SysID                " + str(0)                             + "\n")
else:
    in_file.write("NumMaxEvents         " + str(NumMaxEvents)                  + "\n")
    in_file.write("LumiWeights          " + str(LumiWeights)                   + "\n")
    in_file.write("Universe             " + str(jobs[N][4])                    + "\n")
    in_file.write("SysID_lateral        " + str(jobs[N][3])                    + "\n")
    if analysis != "GEN":
        for systematic in lateral_systematics.keys():
            if lateral_systematics[systematic][0] == jobs[N][3]:
                in_file.write("SysName_lateral " + systematic                      + "\n")
                if len(lateral_systematics[systematic][3]) > 0:
                    SysSubSource = lateral_systematics[systematic][3][int(jobs[N][4]/2)]
                    if SysSubSource[-3:] == "_fc":
                        SysSubSource = SysSubSource[:-3]
                    in_file.write("SysSubSource " + SysSubSource                   + "\n")
    for syst in vertical_systematics.keys():
        in_file.write("SysIDs_vertical  " + str(vertical_systematics[syst][0]) + "\n")
        in_file.write("SysNames_vertical  " + syst                             + "\n")
        
    
#-----DATA AND MC METADATA------------------------------------------------------------
in_file.write("MCmetaFileName       ./Datasets/MC_Metadata.txt"                + "\n")
meta_file = open("Datasets/Data_Metadata.txt")
#meta_file = open("Datasets/Data_Metadata_for_tests.txt") # For bkg cross-sections checks
for line in meta_file:
    no, year, HIPM, data_scale, total_unc, uncorr_unc, fc_unc, fc1718_unc = line.split()
    if( (year == jobs[N][0][1][0:2]) and (HIPM == jobs[N][0][1][6]) ):
        in_file.write("DATA_LUMI             " + data_scale                    + "\n")
        in_file.write("DATA_LUMI_TOTAL_UNC   " + total_unc                     + "\n")
        in_file.write("DATA_LUMI_UNCORR_UNC  " + uncorr_unc                    + "\n")
        in_file.write("DATA_LUMI_FC_UNC      " + fc_unc                        + "\n")
        in_file.write("DATA_LUMI_FC1718_UNC  " + fc1718_unc                    + "\n")
#-------------------------------------------------------------------------------------
    
in_file.write("Show_Timer           "  + str(args.timer)                       + "\n")
in_file.write("Get_Image_in_EPS     "  + str(Get_Image_in_EPS)                 + "\n")
in_file.write("Get_Image_in_PNG     "  + str(Get_Image_in_PNG)                 + "\n")
in_file.write("Get_Image_in_PDF     "  + str(Get_Image_in_PDF)                 + "\n")
    
for cut, value in analysis_parameters.items():
    in_file.write( cut + str(value)                                            + "\n")
    
for corr, value in corrections.items():
    in_file.write( corr + str(value)                                           + "\n")
        
for metaname, metapath in metadata.items():
    metaname = metaname.rstrip()
    meta_year = metaname[-2:]
    job_year = jobs[N][0][0][-2:]
    if( (meta_year == job_year) or (meta_year == "XX") ):
        if( meta_year != "16" ):
            in_file.write( metaname[:-3] + "        " + str(metapath)     + "\n")
        else:
            meta_NORMAL = (metaname[-9:-3] == "NORMAL")
            meta_HIPM = (metaname[-7:-3] == "HIPM")
            if( (not meta_HIPM) and (not meta_NORMAL) ):
                in_file.write( metaname[:-3] + "        " + str(metapath)     + "\n")
            else:
                job_HIPM = ( jobs[N][0][1][6] == "1" )
                if( meta_HIPM and job_HIPM ):
                    in_file.write( metaname[:-8] + "        " + str(metapath)     + "\n")
                if( (not meta_HIPM) and (not job_HIPM) ):
                    in_file.write( metaname[:-10] + "        " + str(metapath)     + "\n")
        
in_file.close()
    
    
#======REMOVE HDF FILE=============================================================================
hdf_file = os.path.join(outpath, selection, jobs[N][0][0] + "_files_" + str(jobs[N][1]) + "_" + str(jobs[N][2]-1), "selection.h5")
removeCommand = "rm " + hdf_file
if os.path.isfile(hdf_file) and jobs[N][3] == 0:  # only for CV
    os.system(removeCommand)
    

#======RUN SELECTION===============================================================================
runCommand = './RunAnalysis ' + ConfigFile
os.system(runCommand)
    
    
#======REMOVE INPUT FILE OF THE SELECTION==========================================================
removeCommand = "rm " + ConfigFile
if os.path.isfile(ConfigFile):
    os.system(removeCommand)


if( (jobs[N][3] == 0) and (jobs[N][4] == 0) ):    
    #======WRITE OUTPUT TXT FILE OF THE SELECTION======================================================
    out_file = open(os.path.join(outpath, selection, jobs[N][0][0] + "_files_" + str(jobs[N][1]) + "_" + str(jobs[N][2]-1), "cutflow.txt"), "a")
    for cut, value in analysis_parameters.items():
        out_file.write( cut + str(value)              + "\n")
    for corr, value in corrections.items():
        out_file.write( corr + str(value)              + "\n")
    for metaname, metapath in metadata.items():
        metaname = metaname.rstrip()
        meta_year = metaname[-2:]
        job_year = jobs[N][0][0][-2:]
        if( (meta_year == job_year) or (meta_year == "XX") ):
            if( meta_year != "16" ):
                out_file.write( metaname[:-3] + "        " + str(metapath)     + "\n")
            else:
                meta_NORMAL = (metaname[-9:-3] == "NORMAL")
                meta_HIPM = (metaname[-7:-3] == "HIPM")
                if( (not meta_HIPM) and (not meta_NORMAL) ):
                    out_file.write( metaname[:-3] + "        " + str(metapath)     + "\n")
                else:
                    job_HIPM = ( jobs[N][0][1][6] == "1" )
                    if( meta_HIPM and job_HIPM ):
                        out_file.write( metaname[:-8] + "        " + str(metapath)     + "\n")
                    if( (not meta_HIPM) and (not job_HIPM) ):
                        out_file.write( metaname[:-10] + "        " + str(metapath)     + "\n")
        
        
    out_file.write("-----------------------------------------------------------------------------------") 
    out_file.close()


    #======PRINT INFO OF THE SELECTION OVER THE DATASET================================================        
    CutflowFile = os.path.join(outpath, selection, jobs[N][0][0] + "_files_" + str(jobs[N][1]) + "_" + str(jobs[N][2]-1), "cutflow.txt")
    out = open(CutflowFile, "r")
    for line in out:
        if( line.rstrip()[:4] == "Time" ):
            print(line.rstrip())
            print('-----------------------------------------------------------------------------------')
            print(' ')
            break
        else:
            print(line.rstrip())
    out.close()

#======PRINT THE CONTENT OF THE HDFf FILE===========================================================
print("=========================HDF5 FILE CONTENT=========================")
if os.path.isfile(hdf_file):

    f = h5py.File(hdf_file, "r")
    variable = []
    var_group = []
    shape = []
    var_mean = []
    var_std = []
    var_NPosEntryPerEvt = []
    for group in f.keys():
        if group == "scalars" or group == "vectors":
            for hdfvar in f[group].keys():
                varname = group+"/"+hdfvar
                var_array = np.array(f[varname])
                variable.append(hdfvar)
                var_group.append(group)
                shape.append(var_array.shape)
                if var_array.size > 0:
                    if group == "scalars":
                        var_mean.append("{:.5f}".format(np.mean(var_array)))
                        var_std.append("{:.5f}".format(np.std(var_array)))
                        var_NPosEntryPerEvt.append("-")
                    elif group == "vectors":
                        if len(var_array[0]) > 0:
                            var_mean.append("{:.5f}".format(np.array(f["metadata/"+hdfvar+"_mean"])))
                            var_std.append("{:.5f}".format(np.array(f["metadata/"+hdfvar+"_std"])))
                            var_NPosEntryPerEvt.append("{:.5f}".format(np.array(f["metadata/"+hdfvar+"_N"])/len(var_array)))
                        else:
                            var_mean.append("empty")
                            var_std.append("empty")
                            var_NPosEntryPerEvt.append("empty")
                else:
                    var_mean.append("empty")
                    var_std.append("empty")
                    var_NPosEntryPerEvt.append("empty")
    df_features = pd.DataFrame({"group": var_group, "feature": variable, "shape": shape, "mean": var_mean, "std": var_std, "NPosEntryPerEvt": var_NPosEntryPerEvt})
    print(df_features)
else:
    print("There is no hdf5 file in the output directory!")
