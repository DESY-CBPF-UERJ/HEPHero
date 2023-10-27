import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import sys
import h5py
import argparse
sys.path.insert(0, '../Datasets')
from Samples import *

from itertools import repeat
import concurrent.futures
import multiprocessing


#======SETUP=======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--selection")
parser.add_argument("-p", "--period")
parser.add_argument("-c", "--cpu")
parser.set_defaults(cpu=None)
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


comb_path = os.path.join(basedir, "datasets")
period_path = os.path.join(comb_path, period)
if( args.apv ):
    period_path = os.path.join(comb_path, "APV_"+period)

if not os.path.exists(comb_path):
    os.makedirs(comb_path)
if not os.path.exists(period_path):
    os.makedirs(period_path)
    
    
jobs_file_name = os.path.join(basedir, "jobs.txt")
json_sys_file_name = os.path.join(basedir, "lateral_systematics.json")
if (not os.path.isfile(jobs_file_name)) or (not os.path.isfile(json_sys_file_name)):
    print('Missing configuration files, execute the runSelection.py using the flag "fix" as in the example below, and then try again!')
    print('python runSelection.py -j 0 --fix')
    sys.exit()
    
with open(json_sys_file_name) as json_sys_file:
    systematics = json.load(json_sys_file)

#reorder sys_sources
list_keys = []
list_0 = []
list_1 = []
list_2 = []
list_3 = []
for sys_source in systematics.keys():
    list_keys.append(sys_source)
    list_0.append(systematics[sys_source][0])
    list_1.append(systematics[sys_source][1])
    list_2.append(systematics[sys_source][2])
    list_3.append(systematics[sys_source][3])

list_0, list_1, list_2, list_3, list_keys = (list(t) for t in zip(*sorted(zip(list_0, list_1, list_2, list_3, list_keys))))

list_values = []
for i in range(len(list_keys)):
    list_values.append([list_0[i], list_1[i], list_2[i], list_3[i]])

systematics = dict(zip(list_keys, list_values))



sample_keys = samples.keys()

def __group_datasets( sample_keys, samples, basedir, period, args_syst, systematics, args_apv, args_debug ):
    if args_debug:
        print("\n ->", sample_keys)
    datasets = sample_keys
    datasets_era = "X"
    datasets_group = "X"
    if len(datasets) > 5:
        datasets_era = datasets[-1]
        datasets_group = datasets[:4]
    preVFP_eras = ["B", "C", "D", "E"]
    postVFP_eras = ["F", "G", "H"]
    if period == "16" and not args_apv and datasets_group == "Data" and (datasets_era in preVFP_eras or 'HIPM_F' in datasets):
        return
    if period == "16" and args_apv and datasets_group == "Data" and datasets_era in postVFP_eras and 'HIPM' not in datasets:
        return
    #print(datasets)
    # Initialize source list (object which will store the systematic histograms)
    if args_syst:
        ds_type = "00"
        with open(jobs_file_name) as jfn:
            for line in jfn:
                info = line.split(" ")
                job_dataset = info[2][3:-5] 
                if( datasets == job_dataset ):
                    ds_type = line.split("', '")[1][2:4]
                    break
        
        source_list = []
        source_idx = 0
        for sys_source in systematics.keys():
            sys_list = systematics[sys_source]
            if( (len(sys_list[2]) == 0) or ((len(sys_list[2]) > 0) and (ds_type in sys_list[2])) ):
                while source_idx <= sys_list[0]:
                    if source_idx < sys_list[0]:
                        source_list.append("empty")
                    elif source_idx == sys_list[0]:
                        universe_list = []
                        for universe in range(sys_list[1]):
                            universe_list.append(0)
                        source_list.append(universe_list)
                    source_idx += 1
    
    
    fpath = os.path.join(period_path, f"{datasets}.h5")
    f_out = h5py.File(fpath, "w")
    
    first_h5 = True
    first_syst = True
    DATA_LUMI = 0
    PROC_XSEC = 0
    SUM_GEN_WGT = 0
    datasets_out_dict = {}
    for dataset in samples[datasets]:
        if args_debug:
            print(dataset)
        dataset_year = dataset.split("_files_")[0]
        dataset_year = dataset_year.split("_")[-1]
        dataset_tag = dataset.split("_"+dataset_year)[0][-3:]
        if (dataset_year == period):
            cutflow = os.path.join(basedir, dataset, "cutflow.txt")
            if os.path.isfile(cutflow):
                with open(cutflow) as cf:
                    for line in cf:
                        if line[:10] == "Luminosity" :
                            DATA_LUMI = float(line.split()[1])
                        if line[:13] == "Cross section" :
                            PROC_XSEC = float(line.split()[2])
                        if line[:17] == "Sum of genWeights" :
                            SUM_GEN_WGT += float(line.split()[3])
                        if line[:17] == "Lumi. Uncertainty" :
                            DATA_LUMI_TOTAL_UNC = float(line.split()[2])
                            DATA_LUMI_UNCORR_UNC = float(line.split()[4])
                            DATA_LUMI_FC_UNC = float(line.split()[5])
                            DATA_LUMI_FC1718_UNC = float(line.split()[6])

                hdf_file = os.path.join(basedir, dataset, "selection.h5")
                f = h5py.File(hdf_file, "r")
                if len(np.array(f["scalars/evtWeight"])) > 0:
                    if first_h5 :
                        for group in f.keys():
                            for hdfvar in f[group].keys():
                                varname = group+"/"+hdfvar
                                if varname[-4:] != "_std" and varname[-5:] != "_mean" and varname[-2:] != "_N":
                                    datasets_out_dict[varname] = np.array(f[varname])
                                if varname[-5:] == "_mean":
                                    N_name = varname[:-5]+"_N"
                                    datasets_out_dict[varname] = np.array(f[varname])*np.array(f[N_name])
                                if varname[-2:] == "_N":
                                    datasets_out_dict[varname] = np.array(f[varname])
                        first_h5 = False
                    else:
                        datasets_new_dict = {}
                        for group in f.keys():
                            for hdfvar in f[group].keys():
                                varname = group+"/"+hdfvar
                                datasets_new_dict[varname] = np.array(f[varname])
                                if group == "vectors":
                                    out_size = len(datasets_out_dict[varname][0])
                                    new_size = len(datasets_new_dict[varname][0])
                                    diff_size = abs(out_size-new_size)
                                    if out_size > new_size:
                                        number_of_events = len(datasets_new_dict[varname])
                                        for i in range(diff_size):
                                            datasets_new_dict[varname] = np.c_[ datasets_new_dict[varname], np.zeros(number_of_events) ]
                                    elif new_size > out_size:
                                        number_of_events = len(datasets_out_dict[varname])
                                        for i in range(diff_size):
                                            datasets_out_dict[varname] = np.c_[ datasets_out_dict[varname], np.zeros(number_of_events) ]
                                if group == "vectors" or group == "scalars":
                                    datasets_out_dict[varname] = np.concatenate((datasets_out_dict[varname],datasets_new_dict[varname]))
                                elif group == "metadata":
                                    if varname[-2:] == "_N":
                                        N_name = varname
                                        mean_name = varname[:-2]+"_mean"
                                        N_new = datasets_new_dict[varname]
                                        mean_new = np.array(f[mean_name])
                                        datasets_out_dict[N_name] += N_new;
                                        datasets_out_dict[mean_name] += N_new*mean_new;
                                    
                        datasets_new_dict.clear()
                    
                #----------------------------------------------------
                # Systematic
                if args_syst:
                    for sys_source in systematics.keys():
                        sys_list = systematics[sys_source]
                        if( (len(sys_list[2]) == 0) or ((len(sys_list[2]) > 0) and (ds_type in sys_list[2])) ):
                            if( (sys_list[0] > 0) and (datasets[:4] == "Data") ): 
                                continue
                            universe_list = []
                            for universe in range(sys_list[1]):
                                sys_file = str(sys_list[0]) + "_" + str(universe) + ".json"
                                with open(os.path.join(basedir, dataset, "Systematics", sys_file)) as json_file:
                                    sys_dict = json.load(json_file)
                                    if first_syst :
                                        source_list[sys_list[0]][universe] = sys_dict.copy()
                                    else:
                                        for variable in sys_dict.keys():
                                            #zipped_Hist = zip(source_list[sys_list[0]][universe][variable]["Hist"], sys_dict[variable]["Hist"])
                                            #New_Hist = [(np.array(x) + np.array(y)).tolist() for (x, y) in zipped_Hist]
                                            #source_list[sys_list[0]][universe][variable]["Hist"] = New_Hist
                                            x = np.array(source_list[sys_list[0]][universe][variable]["Hist"])
                                            y = np.array(sys_dict[variable]["Hist"])
                                            source_list[sys_list[0]][universe][variable]["Hist"] = (x + y).tolist()
                                            #zipped_Unc = zip(source_list[sys_list[0]][universe][variable]["Unc"], sys_dict[variable]["Unc"]) 
                                            #New_Unc = [(np.sqrt(np.array(x)**2 + np.array(y)**2)).tolist() for (x, y) in zipped_Unc]
                                            #source_list[sys_list[0]][universe][variable]["Unc"] = New_Unc
                                            x = np.array(source_list[sys_list[0]][universe][variable]["Unc"])
                                            y = np.array(sys_dict[variable]["Unc"])
                                            source_list[sys_list[0]][universe][variable]["Unc"] = (np.sqrt(x**2 + y**2)).tolist()
                                del sys_dict 
                #---------------------------------------------------
    
                first_syst = False



    if len(datasets_out_dict) > 0:
        for group in f.keys():
            if group == "metadata":
                for hdfvar in f[group].keys():
                    varname = group+"/"+hdfvar
                    if varname[-5:] == "_mean":
                        N_name = varname[:-5]+"_N"
                        std_name = varname[:-5]+"_std"
                        datasets_out_dict[varname] = datasets_out_dict[varname]/datasets_out_dict[N_name];
                        datasets_out_dict[std_name] = 0;
    
    for dataset in samples[datasets]:
        if args_debug:
            print(dataset, "metadata_std")
        if (dataset_year == period):
            hdf_file = os.path.join(basedir, dataset, "selection.h5")
            f = h5py.File(hdf_file, "r")
            if len(np.array(f["scalars/evtWeight"])) > 0:
                for group in f.keys():
                    if group == "metadata":
                        for hdfvar in f[group].keys():
                            varname = group+"/"+hdfvar
                            if varname[-4:] == "_std":
                                N_name = varname[:-4]+"_N"
                                mean_name = varname[:-4]+"_mean"
                                std_name = varname
                                N_new = np.array(f[N_name])
                                mean_new = np.array(f[mean_name])
                                std_new = np.array(f[std_name])
                                mean_all = datasets_out_dict[mean_name]
                                datasets_out_dict[varname] += (N_new-1)*std_new**2 + N_new*(mean_new-mean_all)**2
                                
    
    if len(datasets_out_dict) > 0:
        for group in f.keys():
            if group == "metadata":
                for hdfvar in f[group].keys():
                    varname = group+"/"+hdfvar
                    if varname[-4:] == "_std":
                        N_name = varname[:-4]+"_N"
                        datasets_out_dict[varname] = np.sqrt(datasets_out_dict[varname]/(datasets_out_dict[N_name]-1));


    if PROC_XSEC == 0:
        dataScaleWeight = 1
    else:
        dataScaleWeight = (PROC_XSEC/SUM_GEN_WGT) * DATA_LUMI
    
    
    f_out.create_dataset("metadata/lumiWeight", data=dataScaleWeight)
    f_out.create_dataset("metadata/CrossSection", data=PROC_XSEC)
    f_out.create_dataset("metadata/SumGenWeights", data=SUM_GEN_WGT)
    
    if len(datasets_out_dict) > 0:
        datasets_out_dict["scalars/evtWeight"] = datasets_out_dict["scalars/evtWeight"]*dataScaleWeight
        for group in f.keys():
            for hdfvar in f[group].keys():
                varname = group+"/"+hdfvar
                if varname[-2:] != "_N":
                    f_out.create_dataset(varname, data=datasets_out_dict[varname])
        datasets_out_dict.clear()

    #---SYS--------------------------------------------------------------
    if args_syst:
        output_sys_dict = {}
        if( datasets[:4] == "Data" ): 
            for variable in source_list[0][0].keys():
                output_sys_dict[variable] = source_list[0][0][variable]
        else:
            for isource in range(len(source_list)):
                if source_list[isource] != "empty":
                    for iuniverse in range(len(source_list[isource])):
                        #print(isource, iuniverse)
                        #print(source_list[isource][iuniverse].keys())
                        for variable in source_list[isource][iuniverse].keys():
                            #New_Hist = [(np.array(x)*dataScaleWeight).tolist() for x in source_list[isource][iuniverse][variable]["Hist"]]
                            #source_list[isource][iuniverse][variable]["Hist"] = New_Hist
                            x = np.array(source_list[isource][iuniverse][variable]["Hist"])
                            source_list[isource][iuniverse][variable]["Hist"] = (x*dataScaleWeight).tolist()
                            #New_Unc = [(np.array(x)*dataScaleWeight).tolist() for x in source_list[isource][iuniverse][variable]["Unc"]]
                            #source_list[isource][iuniverse][variable]["Unc"] = New_Unc
                            x = np.array(source_list[isource][iuniverse][variable]["Unc"])
                            source_list[isource][iuniverse][variable]["Unc"] = (x*dataScaleWeight).tolist()
                            output_sys_dict[variable] = source_list[isource][iuniverse][variable]
                            output_sys_dict[variable]["LumiUnc"] = DATA_LUMI_TOTAL_UNC
                            output_sys_dict[variable]["LumiUncorrUnc"] = DATA_LUMI_UNCORR_UNC
                            output_sys_dict[variable]["LumiFCUnc"] = DATA_LUMI_FC_UNC
                            output_sys_dict[variable]["LumiFC1718Unc"] = DATA_LUMI_FC1718_UNC
        
        with open(os.path.join(period_path, f"{datasets}.json"), 'w') as json_file:            
            json.dump(output_sys_dict, json_file)
    
    return "finished"
    
if args.debug:
    for ds in sample_keys:
        __group_datasets( ds, samples, basedir, period, args.syst, systematics, args.apv, args.debug )
else:    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        result = list(tqdm(executor.map(__group_datasets, sample_keys, repeat(samples), repeat(basedir), repeat(period), repeat(args.syst), repeat(systematics), repeat(args.apv), repeat(args.debug)), total=len(sample_keys)))   
        
        
    datasets_path = os.path.join(basedir, "datasets")
    cpCommand = "cp " + os.path.join(basedir, "vertical_systematics.json") + " " + datasets_path
    os.system(cpCommand)
    cpCommand = "cp " + os.path.join(basedir, "lateral_systematics.json") + " " + datasets_path
    os.system(cpCommand)
    cpCommand = "cp " + os.path.join(basedir, args.selection + ".cpp") + " " + datasets_path
    os.system(cpCommand)
