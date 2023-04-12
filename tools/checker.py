import os
import pandas as pd
from tqdm import tqdm
import json
import sys
import h5py
import argparse
sys.path.insert(0, '../Datasets')
from Samples import *

import numpy as np
from itertools import repeat
import concurrent.futures
import multiprocessing


#======SETUP=======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--selection")
parser.add_argument("-p", "--period")
parser.add_argument("-c", "--cpu")
parser.set_defaults(cpu=None)
parser.add_argument("-d", "--datasets")
parser.set_defaults(datasets=None)
parser.add_argument("--syst", dest='syst', action='store_true')
parser.set_defaults(syst=False)
parser.add_argument("--apv", dest='apv', action='store_true')
parser.set_defaults(apv=False)
parser.add_argument("--debug", dest='debug', action='store_true')
parser.set_defaults(debug=False)

args = parser.parse_args()

with open('analysis.txt') as f:
    analysis = f.readline()
    
outpath = os.environ.get("HEP_OUTPATH")
basedir = os.path.join(outpath, analysis, args.selection)
period = str(args.period)

if args.cpu == None:
    cpu_count = multiprocessing.cpu_count()
    if cpu_count <= 2:
        cpu_count = 1
    else:
        cpu_count -= 2
else:
    cpu_count = int(args.cpu)

print('Analysis = ' + analysis)
print('Selection = ' + args.selection)
print('Period = ' + period)
print('Systematics = ' + str(args.syst))
print('APV = ' + str(args.apv))
print('Outpath = ' + basedir)
print('CPUs = ' + str(cpu_count))
print('')


samples = get_samples( basedir, period, args.apv )

TreeName="selection"


jobs_file_name = os.path.join(basedir, "jobs.txt")
json_sys_file_name = os.path.join(basedir, "lateral_systematics.json")
if (not os.path.isfile(jobs_file_name)) or (not os.path.isfile(json_sys_file_name)):
    print('Missing configuration files, execute the runSelection.py using the flag "fix" as in the example below, and then try again!')
    print('python runSelection.py -j 0 --fix')
    sys.exit()

with open(json_sys_file_name) as json_sys_file:
    systematics = json.load(json_sys_file)
#print(systematics)
    
    

def __check_dataset( dataset_list, basedir, period, args_syst, systematics, args_debug ):
    if args_debug:
        print("\n ->", dataset_list)
    job = "None"
    Error_OldFolder = []
    Error_Output = []
    Resubmit_Jobs = []
    dataset = dataset_list
    count_good = 0
    count_bad = 0
    Nentries = 0
    dataset_name = dataset.split("_files_")[0]
    dataset_year = dataset_name.split("_")[-1]
    dataset_tag = dataset.split("_"+dataset_year)[0][-3:]
    if (dataset_year == period):
        #print(dataset)
        control = 0
        with open(jobs_file_name) as f:
            for line in f:
                #if dataset == line[:-1]:
                info = line.split(" ")
                #print(info)
                info_source = info[8].split(",")[0]
                #info_universe = info[9].split("]")[0]
                #print(info_source)
                #print(info_universe)
                job_line = info[2][3:-2] + "_files_" + info[6][:-1] + "_" + str(int(info[7][:-1])-1)
                if( (dataset == job_line) and (info_source == "0") ):
                    control = 1
                    job = line
                    job = "[["+job.split("[[")[1]
                    ds_type = line.split("', '")[1][2:4]
        if control == 0:
            Error_OldFolder.append(dataset)
            return Error_OldFolder, Error_Output, Resubmit_Jobs, count_good, count_bad, Nentries
    
        cutflow = os.path.join(basedir, dataset, "cutflow.txt")
        bad_0_0 = False
        control = 0
        if args_debug:
            print("Checking if cutflow file exist...")
        if os.path.isfile(cutflow):
            if args_debug:
                print("Ok!")
            N_entries = 0
            with open(cutflow) as f:
                for line in f:
                    control += line.count("Time to process the selection")
                    if line[:28] == "Number of entries considered" :
                        N_entries = int(line.split()[4])
            try:
                if args_debug:
                    print("Checking cutflow file content...")
                if control == 1 and N_entries > 0:
                    if args_debug:
                        print("Ok!")
                    hdf_file = os.path.join(basedir, dataset, "selection.h5")
                    if args_debug:
                        print("Checking if hdf file exist...")
                    if os.path.isfile(hdf_file):    
                        if args_debug:
                            print("Ok!")
                        f = h5py.File(hdf_file, "r")
                        if args_debug:
                            print("Checking if hdf file has one or more keys...")
                        if len(f.keys()) >= 1:
                            if args_debug:
                                print("Ok!")
                            Nentries += len(f['scalars/evtWeight'])
                            #print("")
                            if args_syst:     # Check if folders are bad or good
                                sys_control = 0
                                if args_debug:
                                    print("Checking systematic files...")
                                for sys_source in systematics.keys():
                                    sys_list = systematics[sys_source]
                                    if( (len(sys_list[2]) == 0) or ((len(sys_list[2]) > 0) and (ds_type in sys_list[2])) ):
                                        if( (sys_list[0] > 0) and (datasets[:4] == "Data") ):  # Only consider syst. variation in MC
                                            continue
                                        for universe in range(sys_list[1]):
                                            #print(universe) 
                                            sys_file = str(sys_list[0]) + "_" + str(universe) + ".json"
                                            sys_file = os.path.join(basedir, dataset, "Systematics", sys_file)
                                            #print(sys_file)
                                            #print(os.path.isfile(sys_file))
                                            if os.path.isfile(sys_file):
                                                #print(os.stat(sys_file).st_size > 0)
                                                if os.stat(sys_file).st_size > 0:
                                                    with open(sys_file) as json_file:
                                                        sys_dict = json.load(json_file)
                                                        #print(sys_dict.keys())
                                                        if len(sys_dict) == 0: 
                                                            sys_control += 1
                                                else:
                                                    sys_control += 1
                                            else:
                                                sys_control += 1
                                if sys_control == 0:
                                    count_good += 1
                                    if args_debug:
                                        print("Ok!")
                                else: 
                                    count_bad += 1
                                    Error_Output.append(dataset)
                            else:
                                count_good += 1
                            
                        else:
                            count_bad += 1
                            Error_Output.append(dataset)
                            Resubmit_Jobs.append(job)
                            bad_0_0 = True
                    else:
                        count_bad += 1
                        Error_Output.append(dataset)
                        Resubmit_Jobs.append(job)
                        bad_0_0 = True
                else:
                    count_bad += 1
                    Error_Output.append(dataset)
                    Resubmit_Jobs.append(job)
                    bad_0_0 = True
            except Exception as ex:
                print(str(ex), '---->', dataset)
                count_bad += 1
                Error_Output.append(dataset)
                Resubmit_Jobs.append(job)
                bad_0_0 = True
        else:
            count_bad += 1
            Error_Output.append(dataset)
            Resubmit_Jobs.append(job)
            bad_0_0 = True
        
        
        if args_syst:         # Find jobs must be submitted
            for sys_source in systematics.keys():
                if( (sys_source == "CV") and bad_0_0 ):   # Do not consider CV if it is already bad
                    continue
                if( datasets[:4] == "Data" ): 
                    continue
                sys_list = systematics[sys_source]
                if( (len(sys_list[2]) == 0) or ((len(sys_list[2]) > 0) and (ds_type in sys_list[2])) ):
                    for universe in range(sys_list[1]):
                        sys_file = str(sys_list[0]) + "_" + str(universe) + ".json"
                        sys_file = os.path.join(basedir, dataset, "Systematics", sys_file)
                        job_eff = job[:-7] + str(sys_list[0]) + ", " + str(universe) + "]," + "\n" 
                        #print(sys_file)
                        try:
                            if os.path.isfile(sys_file):
                                if os.stat(sys_file).st_size > 0:
                                    with open(sys_file) as json_file:
                                        sys_dict = json.load(json_file)
                                        if len(sys_dict) == 0: 
                                            Resubmit_Jobs.append(job)
                                else:
                                    Resubmit_Jobs.append(job_eff)
                            else:
                                Resubmit_Jobs.append(job_eff)
                        except Exception as ex:
                            Resubmit_Jobs.append(job_eff)

    return Error_OldFolder, Error_Output, Resubmit_Jobs, count_good, count_bad, Nentries
     



Integrity_Jobs = []
Error_OldFolders = []
Error_Output = []
Resubmit_Jobs = []
for datasets in tqdm(samples.keys()):
    if args.datasets != None and datasets != args.datasets:
        continue
    datasets_era = "X"
    datasets_group = "X"
    if len(datasets) > 5:
        datasets_era = datasets[-1]
        datasets_group = datasets[:4]
    preVFP_eras = ["B", "C", "D", "E"]
    postVFP_eras = ["F", "G", "H"]
    if period == "16" and not args.apv and datasets_group == "Data" and (datasets_era in preVFP_eras or 'HIPM_F' in datasets):
        continue
    if period == "16" and args.apv and datasets_group == "Data" and datasets_era in postVFP_eras and 'HIPM' not in datasets:
        continue
    print(datasets)
    jobs_count_good = []
    jobs_count_bad = []
    jobs_Nentries = []
    dataset_list = samples[datasets]
    
    if args.debug:
        for ds in dataset_list:
            __check_dataset( ds, basedir, period, args.syst, systematics, args.debug )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            check_lists_result = list(tqdm(executor.map(__check_dataset, dataset_list, repeat(basedir), repeat(period), repeat(args.syst), repeat(systematics), repeat(args.debug)), total=len(dataset_list)))
        
        for check_lists in check_lists_result:
            if len(check_lists[0]) > 0:
                Error_OldFolders.append(check_lists[0][0])
            if len(check_lists[1]) > 0:    
                Error_Output.append(check_lists[1][0])
            if len(check_lists[2]) > 0:
                Resubmit_Jobs = Resubmit_Jobs + check_lists[2]
            jobs_count_good.append(check_lists[3])
            jobs_count_bad.append(check_lists[4])
            jobs_Nentries.append(check_lists[5])
        
        jobs_count_good_array = np.array(jobs_count_good)
        jobs_count_bad_array = np.array(jobs_count_bad)
        jobs_Nentries_array = np.array(jobs_Nentries)

        Integrity_Jobs.append({
            "Dataset": datasets, 
            "nFolders": len(samples[datasets]), 
            "Good": jobs_count_good_array.sum(), 
            "Bad": jobs_count_bad_array.sum(), 
            "Entries": jobs_Nentries_array.sum()
        })
    
    
if not args.debug:
    if len(Resubmit_Jobs) > 0:
        if( args.apv ):
            file_name = os.path.join("resubmit_APV_" + period + ".txt")
        else:
            file_name = os.path.join("resubmit_" + period + ".txt")
        resubmit_file = open(file_name, "w")
        for i in range(len(Resubmit_Jobs)):
            resubmit_file.write(Resubmit_Jobs[i]) 
    else:
        if( args.apv ):
            file_name = os.path.join("resubmit_APV_" + period + ".txt")
            if os.path.isfile(file_name):
                os.system("mv -f " + file_name + " previous_resubmit_APV_" + period + ".txt" )
        else:
            file_name = os.path.join("resubmit_" + period + ".txt")
            if os.path.isfile(file_name):
                os.system("mv -f " + file_name + " previous_resubmit_" + period + ".txt" )
        
            

    Integrity_Jobs = pd.DataFrame(Integrity_Jobs)

    pd.set_option('display.max_rows', 200)
    print(Integrity_Jobs)

    print("")
    print("====================================================================================================")
    print("List of folders that are not part of the jobs submitted: (remove them!)")
    print(*Error_OldFolders)
    print("====================================================================================================")

    print("")
    print("====================================================================================================")
    print("List of folders with error in the output:")
    print(*Error_Output)
    print("====================================================================================================")
    print("")

    
