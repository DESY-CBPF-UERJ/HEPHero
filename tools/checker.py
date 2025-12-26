import os
import pandas as pd
from tqdm import tqdm
import json
import sys
import h5py
import argparse
import numpy as np
from itertools import repeat
import concurrent.futures
import multiprocessing
import glob


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
parser.add_argument("--debug", dest='debug', action='store_true')
parser.set_defaults(debug=False)

args = parser.parse_args()

with open('analysis.txt') as f:
    analysis = f.readline()


sys.path.insert(0, '../Datasets')
from samples import *

period = str(args.period)
outpath = os.environ.get("HEP_OUTPATH")
user = os.environ.get("USER")
machines = os.environ.get("MACHINES")

if machines == "UERJ":
    basedir = os.path.join("/cms/store/user/", user, "output", analysis, args.selection)
else:
    basedir = os.path.join(outpath, analysis, args.selection)


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
print('Outpath = ' + basedir)
print('CPUs = ' + str(cpu_count))
print('')


jobs_file_name = os.path.join(basedir, "jobs.txt")
json_sys_file_name = os.path.join(basedir, "lateral_systematics.json")
if (not os.path.isfile(jobs_file_name)) or (not os.path.isfile(json_sys_file_name)):
    print('Missing configuration files, execute the runSelection.py using the flag "fix" as in the example below, and then try again!')
    print('python runSelection.py -j 0 --fix')
    sys.exit()


hephero_local_file_name = os.path.join(basedir, "hephero_local.json")
with open(hephero_local_file_name) as hephero_local_file:
    hephero_local = json.load(hephero_local_file)
    machines_origin = hephero_local["MACHINES"]
    user_origin = hephero_local["USER"]
    hephero_path_origin = hephero_local["HEPHERO_PATH"]

samples = get_samples( analysis, basedir, period )

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
    dataset = dataset_list # It is a specific folder
    count_good = 0
    count_bad = 0
    Nentries = 0
    dataset_name = dataset.split("_files_")[0]
    dataset_period = dataset_name.split("_")[-2]+"_"+dataset_name.split("_")[-1]
    if (dataset_period == period):
        #print("dataset", dataset)
        control = 0
        with open(jobs_file_name) as f:
            for line in f:
                cross_section_unc_string = line.split("', ")[3].split(", '")[1]
                line = line.replace("'"+cross_section_unc_string+"'", cross_section_unc_string)
                #print(line)
                info = line.split(" ")
                #print(info)
                info_source = info[10].split(",")[0]
                #print("info_source",info_source)
                job_line = info[2][3:-2] + "_files_" + info[8][:-1] + "_" + str(int(info[9][:-1])-1)
                #print("job_line", job_line)
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


job_samples = []
with open(jobs_file_name) as f:
    for line in f:
        info = line.split(" ")
        job_period = info[2][-6:-2]
        job_dataset = info[2][3:-7]
        job_samples.append(job_dataset)
job_samples = list(dict.fromkeys(job_samples))

extra_samples = get_samples( analysis, basedir, period )
for extra_sample in extra_samples.keys():
    if extra_sample not in job_samples:
        job_samples.append(extra_sample)

Integrity_Jobs = []
Error_OldFolders = []
Error_Output = []
Resubmit_Jobs = []
for datasets in tqdm(job_samples):
    if datasets in samples:
        if args.datasets != None and datasets != args.datasets:
            continue
        datasets_era = "X"
        datasets_group = "X"
        if len(datasets) > 5:
            datasets_era = datasets[-1]
            datasets_group = datasets[:4]
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

            jobs_count_missing = 0
            with open(jobs_file_name) as f:
                for line in f:
                    cross_section_unc_string = line.split("', ")[3].split(", '")[1]
                    line = line.replace("'"+cross_section_unc_string+"'", cross_section_unc_string)
                    info = line.split(" ")
                    job_period = info[2][-6:-2]
                    job_dataset = info[2][3:-7]
                    if (job_period == period) and (job_dataset == datasets):
                        info_source = info[10].split(",")[0]
                        #print("info_sorce", info_source)
                        job_line = info[2][3:-2] + "_files_" + info[8][:-1] + "_" + str(int(info[9][:-1])-1)
                        #print("line", line)
                        #print("line2", line.split("', ")[3].split(", '")[1])
                        #cross_section_unc_string = line.split("', ")[3].split(", '")[1]
                        #line = line.replace("'"+cross_section_unc_string+"'", cross_section_unc_string)
                        #print("line", line)
                        if( (job_line not in dataset_list) and (info_source == "0") ):
                            job = "[["+line.split("[[")[1]
                            print("job", job)
                            Resubmit_Jobs.append(job)
                            jobs_count_missing += 1

            Integrity_Jobs.append({
                "Dataset": datasets,
                "nFolders": len(samples[datasets])+jobs_count_missing,
                "Good": jobs_count_good_array.sum(),
                "Bad": jobs_count_bad_array.sum(),
                "Missing": jobs_count_missing,
                "Entries": jobs_Nentries_array.sum()
            })
    else:
        jobs_count_missing = 0
        with open(jobs_file_name) as f:
            for line in f:
                cross_section_unc_string = line.split("', ")[3].split(", '")[1]
                line = line.replace("'"+cross_section_unc_string+"'", cross_section_unc_string)
                info = line.split(" ")
                job_period = info[2][-6:-2]
                job_dataset = info[2][3:-7]
                info_source = info[10].split(",")[0]
                if (job_period == period) and (job_dataset == datasets) and ( info_source == "0" ):
                    job = "[["+line.split("[[")[1]
                    Resubmit_Jobs.append(job)
                    jobs_count_missing += 1

        Integrity_Jobs.append({
            "Dataset": datasets,
            "nFolders": jobs_count_missing,
            "Good": 0,
            "Bad": 0,
            "Missing": jobs_count_missing,
            "Entries": 0
        })


if not args.debug:
    print("Resubmit_Jobs", len(Resubmit_Jobs))
    if len(Resubmit_Jobs) > 0:
        file_name = os.path.join("resubmit_" + period + ".txt")
        resubmit_file = open(file_name, "w")
        for i in range(len(Resubmit_Jobs)):
            resubmit_file.write(Resubmit_Jobs[i])
        resubmit_file.close()
    #else:
    #    file_name = os.path.join("resubmit_" + period + ".txt")
    #    if os.path.isfile(file_name):
    #        os.system("mv -f " + file_name + " previous_resubmit_" + period + ".txt" )

    Integrity_Jobs = pd.DataFrame(Integrity_Jobs)
    Integrity_Jobs = Integrity_Jobs[Integrity_Jobs["nFolders"] > 0]

    pd.set_option('display.max_rows', 200)
    print("")
    print(Integrity_Jobs)

    print("")
    print("====================================================================================================")
    print("List of folders that are not part of the jobs submitted:")
    print(*Error_OldFolders)
    print("====================================================================================================")

    print("")
    print("====================================================================================================")
    print("List of folders with error in the output:")
    print(*Error_Output)
    print("====================================================================================================")
    print("")


    resubmit_list = glob.glob("resubmit_*")
    if machines == "UERJ" and len(resubmit_list) > 0:
        if machines_origin == "CERN":
            command = "scp resubmit_* " + user_origin+"@lxplus.cern.ch:"+hephero_path_origin+"/tools"
        elif machines_origin == "CMSC":
            command = "scp resubmit_* " + user_origin+"@login.uscms.org:"+hephero_path_origin+"/tools"

        if machines_origin != "UERJ":
            print("resubmit list:", resubmit_list)
            print(command)
            os.system(command)
            print("")


