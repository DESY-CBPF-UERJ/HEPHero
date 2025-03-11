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
analysis = 'AP_Template_R3'
treeName = 'Events'
LumiWeights = 1

analysis_parameters = {
}

corrections = {  # 0-don't apply, 1-apply
}


#-------------------------------------------------------------------------------------
# Systematics
#-------------------------------------------------------------------------------------
lateral_systematics = {
'CV': [0, 1, [], []],
}

vertical_systematics = {
}


#-------------------------------------------------------------------------------------
# Jobs setup
#-------------------------------------------------------------------------------------
NumMaxEvents = -1
NumFilesPerJob_Data = 1
NumFilesPerJob_Signal = 50
NumFilesPerJob_Bkg = 5


#-------------------------------------------------------------------------------------
# Datasets
#-------------------------------------------------------------------------------------
sys.path.insert(0, 'AP_Template_R3/Datasets')
from Signal import *
from Bkg import *
from Data import *
datasets = []

datasets.extend(Data_MET_0_22)
datasets.extend(Signal_0_22)


#-------------------------------------------------------------------------------------
# Metadata
#-------------------------------------------------------------------------------------
metadata = {
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
print('')
print('Analysis = ' + analysis)
print('Job ID = ' + str(args.job))
print('Timer = ' + str(args.timer))
print('Proxy = ' + args.proxy)   
print('Check = ' + str(args.check))
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
    outpath = analysis+"/ana/local_output"
    redirector = "None"
    N = 0
    
    if analysis == "GEN":
        if args.check == 0:
            datasets = [["H7_test", '1600000', analysis+"/Datasets/test.hepmc", 0],]
            #datasets = [["H7_cms", '1600000', analysis+"/Datasets/InterfaceMatchboxTest-S123456790_cms.hepmc", 0],]
            #datasets = [["H7_dell", '1600000', analysis+"/Datasets/InterfaceMatchboxTest-S123456790_dell.hepmc", 0],]
            #datasets = [["H7_ttbar_tests", '1600000', "/home/gcorreia/cernbox/ttbar_test/LHC-Matchbox.hepmc", 0],]
        if args.check >= 1:
            sys.exit("There are only 1 test from 0 to 0.")
    else:
        if args.check == 0:
            datasets = [["Signal_test_1_16",                  '1600001', analysis+"/Datasets/signal.root", 0],]
        if args.check == 1:
            datasets = [["TTToSemiLeptonic_test_0_16",        '1600000', analysis+"/Datasets/bkg.root",  0],]
        if args.check == 2:
            datasets = [["TTToSemiLeptonic_test_1_16",        '1600001', analysis+"/Datasets/bkg.root",  0],]
        if args.check == 3:
            datasets = [["TTToSemiLeptonic_test_0_17",        '1700000', analysis+"/Datasets/bkg.root",  0],]
        if args.check == 4:
            datasets = [["TTToSemiLeptonic_test_0_18",        '1800000', analysis+"/Datasets/bkg.root",  0],]
        if args.check == 5:
            datasets = [["TTToSemiLeptonic_test_0_22",        '2200000', analysis+"/Datasets/bkg.root",  0],]
        if args.check == 6:
            datasets = [["TTToSemiLeptonic_test_0_23",        '2300000', analysis+"/Datasets/bkg.root",  0],]
        if args.check == 7:
            datasets = [["Data_DoubleMu_test_H_1_16",         '1600001', analysis+"/Datasets/data.root",   0],]
        if args.check >= 8:
            sys.exit("There are only 8 tests from 0 to 7.")
    
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
            datasets_short = [job[0][2].split("Files/")[1] for job in jobs]
            for i in range(len(jobs)):
                #print(i, jobs[i])
                print(i, [jobs[i][0][0], jobs[i][0][1], datasets_short[i]], [jobs[i][0][3], str(jobs[i][1])+"-"+str(jobs[i][2])], [jobs[i][3], jobs[i][4]])
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
                if (machines != "CERN") and (machines != "CMSC"):
                    file_input = open(dataset[2], 'r')
                    lines = file_input.readlines()
                    lines = [x.strip() for x in lines]
                    NumFiles = 0
                    for line in lines:
                        if machines == "DESY":
                            file_path = "/pnfs/desy.de/cms/tier2/" + line
                        if machines == "UERJ":
                            file_path = "/cms/" + line
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
            datasets_short = [job[0][2].split("Files/")[1] for job in jobs]
            for i in range(len(jobs)):
                print(str(i)+" ["+jobs[i][0][0]+", "+jobs[i][0][1]+", "+datasets_short[i]+"], ["+str(jobs[i][0][3])+", "+str(jobs[i][1])+"-"+str(jobs[i][2])+"], ["+str(jobs[i][3])+", "+str(jobs[i][4])+"]")
            sys.exit("")
        if N == -3:
            print("")
            for i in range(len(empty_text_files)):
                print(empty_text_files[i])
            print("")
            sys.exit("There are " + str(len(empty_text_files)) + " empty text files")
        if (machines != "CERN") and (machines != "CMSC"):
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
    copyfile(analysis+"/ana/"+selection+".cpp", outpath+'/'+selection+"/"+selection+".cpp")
    
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
ConfigFile = analysis+"/Metadata/ConfigFile_" + jobs[N][0][0] + "_" + str(jobs[N][1]) + "_" + str(jobs[N][2]-1) + "_" + str(jobs[N][3]) + "_" + str(jobs[N][4]) + ".txt"
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
        if (machines != "CERN") and (machines != "CMSC") and (analysis != "GEN"):
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
in_file.write("MCmetaFileName       ./"+analysis+"/Datasets/MC_Metadata.txt"                + "\n")
meta_file = open(analysis+"/Datasets/Data_Metadata.txt")
#meta_file = open(analysis+"/Datasets/Data_Metadata_for_tests.txt") # For bkg cross-sections checks
first_line = True
for line in meta_file:
    if first_line:
        tags_split_unc = [unc_tag[4:-3] for unc_tag in line.split()[5:]]
        first_line = False
    year = line.split()[1]
    dti = line.split()[2]
    data_scale = line.split()[3]
    total_unc = line.split()[4]
    values_split_unc = line.split()[5:]
    if( (year == jobs[N][0][1][0:2]) and (dti == jobs[N][0][1][6]) ):
        in_file.write("DATA_LUMI             " + data_scale                    + "\n")
        in_file.write("DATA_LUMI_TOTAL_UNC   " + total_unc                     + "\n")
        for i in range(len(values_split_unc)):
            in_file.write("DATA_LUMI_TAGS_UNC    " + tags_split_unc[i]         + "\n")
            in_file.write("DATA_LUMI_VALUES_UNC  " + values_split_unc[i]       + "\n")
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
    meta_dti = metaname[-4:-3]
    job_year = jobs[N][0][0][-2:]
    job_dti = jobs[N][0][0][-4:-3]
    if( (meta_year == "XX") or ((meta_dti == "X") and (meta_year == job_year)) or ((meta_dti == job_dti) and (meta_year == job_year)) ):
        in_file.write( metaname[:-5] + "        " + str(metapath)     + "\n")

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
        meta_dti = metaname[-4:-3]
        job_year = jobs[N][0][0][-2:]
        job_dti = jobs[N][0][0][-4:-3]
        if( (meta_year == "XX") or ((meta_dti == "X") and (meta_year == job_year)) or ((meta_dti == job_dti) and (meta_year == job_year)) ):
            out_file.write( metaname[:-5] + "        " + str(metapath)     + "\n")
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
