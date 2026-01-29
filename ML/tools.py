import sys
import pandas as pd
import os
import concurrent.futures as cf
from operator import itemgetter
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as pat
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
plt.style.use('classic')
from tqdm import tqdm
import json
import h5py
import math

seed = 16
import numpy as np
numpy_random = np.random.RandomState(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
import torch.nn as nn
from custom_opts.ranger import Ranger

from models.NN.model import *
from models.PNN.model import *
from models.APNN.model import *
#from models.APSNN import *
#from models.PNET import *
#from models.M2CNN import *
#from models.DANN import *

"""
-> model training minimize classification and domain at the same time (affect all weights)
-> The weights of the domain branch are recovered
-> domain model training minimize domain (classification branch weights are not affected)
-> The weights of the comum branch are recovered (the weights of the classification model are effectivily updated in the "model training")

-> The target samples don't influence the classification training (sample_weights)
-> During the model training, the weights are modified to improved the accuaracy of the domain and classification prediction
-> During the domain model training, the weights are modified to make the network doesn't be able to distinguish the domains
"""


#==================================================================================================
def get_sample(basedir, period, classes, n_signal, train_frac, load_size, load_it, reweight_info, features=[], vec_features=[], verbose=False, normalization_method="evtsum"):

    has_weights = False
    if len(reweight_info) > 0:
        reweight_vars = [reweight_info[i][0] for i in range(len(reweight_info))]
        reweight_limits = [reweight_info[i][1] for i in range(len(reweight_info))]
        if reweight_vars[-1] == "var_weights":
            var_weights = reweight_limits[-1]
            reweight_vars = reweight_vars[:-1]
            reweight_limits = reweight_limits[:-1]
            has_weights = True
        else:
            var_weights = {}

    class_names = []
    class_labels = []
    class_colors = []
    control = True

    if not verbose:
            print("Loading datasets entries")

    for class_key in classes: # for each class
        info_list = []

        if len(reweight_info) > 0 and not has_weights:
            if len(reweight_vars) == 1:
                var_weights[class_key] = np.ones((2,len(reweight_limits[0])-1))
            elif len(reweight_vars) == 2:
                var_weights[class_key] = np.ones((2,len(reweight_limits[0])-1,len(reweight_limits[1])-1))
            elif len(reweight_vars) == 3:
                var_weights[class_key] = np.ones((2,len(reweight_limits[0])-1,len(reweight_limits[1])-1,len(reweight_limits[2])-1))

        if class_key[:14] == "Signal_samples":
            class_name = classes[class_key][0][n_signal]
            class_label = classes[class_key][0][n_signal]
            input_list = [classes[class_key][0][n_signal]]
        else:
            class_name = class_key
            class_label = classes[class_key][3]
            input_list = classes[class_name][0]
        class_color = classes[class_key][4]

        mode = classes[class_key][1]
        combination = classes[class_key][2]

        #print(" ")
        if verbose:
            print("Loading datasets entries of class", class_key)
        #print("load_it", load_it)

        datasets_dir = os.path.join(basedir, period)
        datasets_abspath = [(f, os.path.join(datasets_dir, f)) for f in os.listdir(datasets_dir)]

        #=================================================================================
        datasets_length = []
        datasets_evtWsum = []
        datasets_names = []
        n_datasets = 0
        for dataset, abspath in datasets_abspath: # for each dataset in a class
            dataset_name = dataset.split(".")[0]

            if dataset.endswith(".h5") and dataset_name in input_list:
                with h5py.File(abspath) as f:
                    datasets_length.append(len(np.array(f["scalars/evtWeight"]))) # number of entries
                    datasets_evtWsum.append(np.array(f["scalars/evtWeight"]).sum()) # sum of weights
                    datasets_names.append(dataset_name)
                n_datasets += 1

        if combination == "balanced" or combination == "equal":
            datasets_frac = np.ones(n_datasets)*(1./n_datasets)
        elif combination == "evtsum":
            datasets_evtWsum = np.array(datasets_evtWsum)
            total_evtWsum = datasets_evtWsum.sum()
            datasets_frac = datasets_evtWsum/total_evtWsum

        #=================================================================================
        class_load_size = int(load_size/len(classes))

        datasets_entries = datasets_frac*class_load_size
        datasets_pdf = datasets_entries/datasets_entries.sum()
        datasets_entries = np.array([ int(i) if int(i) >= 2 else 2 for i in datasets_entries])
        datasets_assigned_entries = datasets_entries.copy()
        datasets_entries = np.minimum(datasets_entries, datasets_length)

        for itry in range(3):
            datasets_needed_entries = datasets_length - datasets_entries
            total_remaining_entries = datasets_assigned_entries.sum() - datasets_entries.sum()
            if total_remaining_entries > 0 and datasets_needed_entries.sum() > 0:
                datasets_pdf = np.array([datasets_pdf[i] if datasets_needed_entries[i] > 0 else 0 for i in range(len(datasets_pdf))])
                datasets_pdf = datasets_pdf/datasets_pdf.sum()
                datasets_provided_entries = total_remaining_entries*datasets_pdf
                datasets_provided_entries = [int(i) for i in datasets_provided_entries]

                datasets_entries = np.array([ datasets_entries[i]+datasets_provided_entries[i] if datasets_needed_entries[i] >= datasets_provided_entries[i] else datasets_entries[i]+datasets_needed_entries[i] for i in range(len(datasets_entries))])

        datasets_train_entries = train_frac*datasets_entries
        datasets_train_entries = np.array([ int(i) for i in datasets_train_entries])
        datasets_test_entries = datasets_entries - datasets_train_entries

        datasets_nSlices = np.array([int(datasets_length[i]/datasets_entries[i]) if datasets_entries[i] > 0 else 0 for i in range(len(datasets_length))])

        datasets_slices = load_it%datasets_nSlices
        #print("datasets_slices", datasets_slices)

        datasets_train_limits = [[datasets_slices[i]*datasets_entries[i], datasets_train_entries[i]+datasets_slices[i]*datasets_entries[i]] for i in range(len(datasets_slices))]
        datasets_test_limits = [[datasets_train_entries[i]+datasets_slices[i]*datasets_entries[i], (datasets_slices[i]+1)*datasets_entries[i]] for i in range(len(datasets_slices))]

        #print("datasets_train_limits", datasets_train_limits)
        #print("datasets_test_limits", datasets_test_limits)

        if verbose:
            info_df = pd.DataFrame({
                'Dataset Name': datasets_names,
                'Available': datasets_length,
                'Loaded': datasets_entries,
                'Train': datasets_train_entries,
                'Test': datasets_test_entries,
                'nSlices': datasets_nSlices,
            })
            print(info_df)
            print(" ")

        #=================================================================================

        for it in range(2):
            datasets = {}
            datasets_vec = {}
            ids = 0
            #for dataset, abspath in tqdm(datasets_abspath):
            for dataset, abspath in datasets_abspath:
                dataset_name = dataset.split(".")[0]
                if dataset.endswith(".h5") and dataset_name in input_list:

                    # Getting % usage of virtual_memory ( 3rd field)
                    #print('RAM memory % used:', psutil.virtual_memory()[2])
                    # Getting usage of virtual_memory in GB ( 4th field)
                    #print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

                    variables_dict = {}
                    with h5py.File(abspath) as f:
                        if "scalars" in f.keys():
                            group = "scalars"
                            for variable in f[group].keys():
                                if len(features) == 0 or variable in features:
                                    if it == 0:
                                        variables_dict[variable] = np.array(f[group+"/"+variable][datasets_train_limits[ids][0]:datasets_train_limits[ids][1]])
                                    elif it == 1:
                                        variables_dict[variable] = np.array(f[group+"/"+variable][datasets_test_limits[ids][0]:datasets_test_limits[ids][1]])
                            datasets[dataset_name] = variables_dict
                        else:
                            print("Warning: Dataset " + dataset_name + " is empty!")

                    if len(vec_features) > 0:
                        variables_dict = {}
                        with h5py.File(abspath) as f:
                            if "vectors" in f.keys():
                                group = "vectors"
                                for variable in f[group].keys():
                                    if variable in vec_features:
                                        if it == 0:
                                            variables_dict[variable] = np.array(f[group+"/"+variable][datasets_train_limits[ids][0]:datasets_train_limits[ids][1]])
                                        elif it == 1:
                                            variables_dict[variable] = np.array(f[group+"/"+variable][datasets_test_limits[ids][0]:datasets_test_limits[ids][1]])
                                datasets_vec[dataset_name] = variables_dict
                            else:
                                print("Warning: Dataset " + dataset_name + " is empty!")

                    if len(datasets[dataset_name]["evtWeight"]) > 0:
                        if combination == "equal":
                            datasets[dataset_name]["evtWeight"] = datasets[dataset_name]["evtWeight"]/datasets[dataset_name]["evtWeight"].sum()
                        elif combination == "evtsum" or combination == "balanced":
                            if datasets[dataset_name]["evtWeight"].sum() != 0:
                                ds_factor = datasets_evtWsum[ids]/datasets[dataset_name]["evtWeight"].sum()
                                datasets[dataset_name]["evtWeight"] = datasets[dataset_name]["evtWeight"]*ds_factor
                            else:
                                datasets[dataset_name]["evtWeight"] = datasets[dataset_name]["evtWeight"]*0.

                    ids += 1

            #==========================================================================
            check_list = [True if len(datasets[input_name]["evtWeight"]) > 0 else False for input_name in input_list]
            #print("check_list", check_list)

            if len(input_list) > 0:
                join_datasets(datasets, class_name, input_list, check_list, mode="scalars", combination=combination)
                if len(vec_features) > 0:
                    join_datasets(datasets_vec, class_name, input_list, check_list, mode="vectors", combination=combination)
            #==========================================================================

            ikey = 0
            for key in classes:
                if key == class_key:
                    break
                ikey += 1

            n_entries = len(datasets[class_name]['evtWeight'])
            p_idx = numpy_random.permutation(n_entries)

            dataset = {}
            for variable in datasets[class_name].keys():
                dataset[variable] = datasets[class_name][variable][p_idx]
                datasets[class_name][variable] = 0
            del datasets

            dataset["class"] = np.ones(n_entries)*ikey
            dataset['mvaWeight'] = dataset['evtWeight']/dataset['evtWeight'].sum()

            dataset_vec = {}
            if len(vec_features) > 0:
                for variable in datasets_vec[class_name].keys():
                    dataset_vec[variable] = datasets_vec[class_name][variable][p_idx]
                    datasets_vec[class_name][variable] = 0
                del datasets_vec

            
            #==========================================================================
            # REWEIGHT PART
            if len(reweight_info) > 0:
                if len(reweight_vars) == 1:
                    split1 = np.array(reweight_limits[0])
                    var1 = reweight_vars[0]
                    data_var = {var1: dataset[var1], 'mvaWeight': dataset['mvaWeight']}
                    data_var = pd.DataFrame.from_dict(data_var)
                    for j in range(len(split1)-1):
                        if has_weights:
                            fac = var_weights[class_key][it,j]
                        else:
                            bin_Wsum = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1]))]['mvaWeight'].sum()
                            bin_base = split1[j+1]-split1[j]
                            if normalization_method == "evtsum":
                                fac = 1/bin_Wsum
                            elif normalization_method == "area":
                                fac = 1/(bin_Wsum*bin_base)
                            if math.isnan(fac):
                                fac = 1
                            var_weights[class_key][it,j] = fac
                        data_var.loc[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])), 'mvaWeight'] = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1]))]['mvaWeight']*fac

                    dataset['mvaWeight'] = np.array(data_var['mvaWeight']/data_var['mvaWeight'].sum())
                    del data_var

                elif len(reweight_vars) == 2:
                    split1 = reweight_limits[0]
                    var1 = reweight_vars[0]
                    split2 = reweight_limits[1]
                    var2 = reweight_vars[1]
                    data_var = {var1: dataset[var1], var2: dataset[var2], 'mvaWeight': dataset['mvaWeight']}
                    data_var = pd.DataFrame.from_dict(data_var)
                    for j in range(len(split1)-1):
                        for i in range(len(split2)-1):
                            if has_weights:
                                fac = var_weights[class_key][it,j,i]
                            else:
                                bin_Wsum = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1]))]['mvaWeight'].sum()
                                bin_base = (split1[j+1]-split1[j])*(split2[i+1]-split2[i])
                                if normalization_method == "evtsum":
                                    fac = 1/bin_Wsum
                                elif normalization_method == "area":
                                    fac = 1/(bin_Wsum*bin_base)
                                if math.isnan(fac):
                                    fac = 1
                                var_weights[class_key][it,j,i] = fac
                            data_var.loc[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1])), 'mvaWeight'] = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1]))]['mvaWeight']*fac

                    dataset['mvaWeight'] = np.array(data_var['mvaWeight']/data_var['mvaWeight'].sum())
                    del data_var

                elif len(reweight_vars) == 3:
                    split1 = reweight_limits[0]
                    var1 = reweight_vars[0]
                    split2 = reweight_limits[1]
                    var2 = reweight_vars[1]
                    split3 = reweight_limits[2]
                    var3 = reweight_vars[2]
                    data_var = {var1: dataset[var1], var2: dataset[var2], var3: dataset[var3], 'mvaWeight': dataset['mvaWeight']}
                    data_var = pd.DataFrame.from_dict(data_var)
                    for j in range(len(split1)-1):
                        for i in range(len(split2)-1):
                            for k in range(len(split3)-1):
                                if has_weights:
                                    fac = var_weights[class_key][it,j,i,k]
                                else:
                                    bin_Wsum = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1])) & ((data_var[var3] >= split3[k]) & (data_var[var3] < split3[k+1]))]['mvaWeight'].sum()
                                    bin_base = (split1[j+1]-split1[j])*(split2[i+1]-split2[i])*(split3[k+1]-split3[k])
                                    if normalization_method == "evtsum":
                                        fac = 1/bin_Wsum
                                    elif normalization_method == "area":
                                        fac = 1/(bin_Wsum*bin_base)
                                    if math.isnan(fac):
                                        fac = 1
                                    var_weights[class_key][it,j,i,k] = fac
                                data_var.loc[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1])) & ((data_var[var3] >= split3[k]) & (data_var[var3] < split3[k+1])), 'mvaWeight'] = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1])) & ((data_var[var3] >= split3[k]) & (data_var[var3] < split3[k+1]))]['mvaWeight']*fac

                    dataset['mvaWeight'] = np.array(data_var['mvaWeight']/data_var['mvaWeight'].sum())
                    del data_var

                if not has_weights:
                    if it == 0:
                        print("Train weights for "+class_key+":")
                    elif it == 1:
                        print("Test weights for "+class_key+":")
                    #print("reweight_vars", reweight_vars)
                    #print("reweight_limits", reweight_limits)
                    print(var_weights[class_key][it])
                    print(" ")
            #dataset['mvaWeight'] = dataset['mvaWeight']/dataset['mvaWeight'].sum()


            #==========================================================================
            if it == 0:
                dataset_train = dataset.copy()
                dataset_vec_train = dataset_vec.copy()
            elif it == 1:
                dataset_test = dataset.copy()
                dataset_vec_test = dataset_vec.copy()
            del dataset, dataset_vec

        if len(reweight_info) > 0 and not has_weights:
            var_weights[class_key][0] = (var_weights[class_key][0] + var_weights[class_key][1])*0.5
            var_weights[class_key][1] = var_weights[class_key][0]
            #print("var_weights_after", var_weights[class_key][0])
            #print("var_weights_after", var_weights[class_key][1])

        class_names.append(class_name)
        class_labels.append(class_label)
        class_colors.append(class_color)

        if control:
            ds_full_train = dataset_train.copy()
            ds_full_test = dataset_test.copy()
            vec_full_train = dataset_vec_train.copy()
            vec_full_test = dataset_vec_test.copy()
            control = False
        else:
            for variable in ds_full_train.keys():
                ds_full_train[variable] = np.concatenate((ds_full_train[variable], dataset_train[variable]), axis=0)
                ds_full_test[variable] = np.concatenate((ds_full_test[variable], dataset_test[variable]), axis=0)
            for variable in vec_full_train.keys():
                #print(variable)
                #print(vec_full_train[variable].shape)
                #print(dataset_vec_train[variable].shape)

                out_size = len(vec_full_train[variable][0])
                dataset_size = len(dataset_vec_train[variable][0])
                diff_size = abs(out_size-dataset_size)
                if out_size > dataset_size:
                    number_of_events = len(dataset_vec_train[variable])
                    for i in range(diff_size):
                        dataset_vec_train[variable] = np.c_[ dataset_vec_train[variable], np.zeros(number_of_events) ]
                elif dataset_size > out_size:
                    number_of_events = len(vec_full_train[variable])
                    for i in range(diff_size):
                        vec_full_train[variable] = np.c_[ vec_full_train[variable], np.zeros(number_of_events) ]
                vec_full_train[variable] = np.concatenate((vec_full_train[variable], dataset_vec_train[variable]), axis=0)

                out_size = len(vec_full_test[variable][0])
                dataset_size = len(dataset_vec_test[variable][0])
                diff_size = abs(out_size-dataset_size)
                if out_size > dataset_size:
                    number_of_events = len(dataset_vec_test[variable])
                    for i in range(diff_size):
                        dataset_vec_test[variable] = np.c_[ dataset_vec_test[variable], np.zeros(number_of_events) ]
                elif dataset_size > out_size:
                    number_of_events = len(vec_full_test[variable])
                    for i in range(diff_size):
                        vec_full_test[variable] = np.c_[ vec_full_test[variable], np.zeros(number_of_events) ]
                vec_full_test[variable] = np.concatenate((vec_full_test[variable], dataset_vec_test[variable]), axis=0)

        if verbose:
            print(" ")
            print(" ")

    if len(reweight_info) > 0 and not has_weights:
        reweight_info.append(["var_weights", var_weights])
    del dataset_train, dataset_test, dataset_vec_train, dataset_vec_test
    
    return ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, class_colors, reweight_info


#==================================================================================================
def check_scalars(train_data, variables, var_names, var_use, var_bins, class_names, class_labels, class_colors, plots_outpath):

    train_data = pd.DataFrame.from_dict(train_data)

    for ivar in range(len(variables)):
        if var_bins[ivar] is not None:
            fig1 = plt.figure(figsize=(9,5))
            gs1 = gs.GridSpec(1, 1)
            #==================================================
            ax1 = plt.subplot(gs1[0])
            #==================================================
            var = variables[ivar]
            bins = var_bins[ivar]
            for ikey in range(len(class_names)):
                yHist, errHist = tools.step_plot( ax1, var, train_data[train_data["class"] == ikey], label=class_labels[ikey]+" (train)", color=class_colors[ikey], weight="mvaWeight", bins=bins, error=True )
                #print(variables[ivar], class_names[ikey])
                #print("yHist", np.round(yHist,6).tolist())
                #print("errHist", np.round(errHist,6).tolist())

            ax1.set_xlabel(var_names[ivar], size=14, horizontalalignment='right', x=1.0)
            ax1.set_ylabel("Events normalized", size=14, horizontalalignment='right', y=1.0)

            ax1.tick_params(which='major', length=8)
            ax1.tick_params(which='minor', length=4)
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.yaxis.set_minor_locator(AutoMinorLocator())
            ax1.spines['bottom'].set_linewidth(1)
            ax1.spines['top'].set_linewidth(1)
            ax1.spines['left'].set_linewidth(1)
            ax1.spines['right'].set_linewidth(1)
            ax1.margins(x=0)
            ax1.legend(numpoints=1, ncol=2, prop={'size': 10.5}, frameon=False)

            plt.subplots_adjust(left=0.09, bottom=0.115, right=0.97, top=0.95, wspace=0.18, hspace=0.165)
            plt.savefig(os.path.join(plots_outpath, var + '_check_.png'), dpi=400)
            plt.savefig(os.path.join(plots_outpath, var + '_check_.pdf'))
            plt.close()




#==================================================================================================
def join_datasets(ds, new_name, input_list, check_list, mode="scalars", combination="evtsum"):

    datasets_list = []
    for i in range(len(input_list)):
        if check_list[i]:
            if mode == "scalars" and combination == "equal":
                ds[input_list[i]]["evtWeight"] = ds[input_list[i]]["evtWeight"]/ds[input_list[i]]["evtWeight"].sum()
            datasets_list.append(ds[input_list[i]])

    good_list = False
    if mode == "normal":
        ds[new_name] = pd.concat(datasets_list).reset_index(drop=True)
        good_list = True
    elif mode == "syst":
        ds[new_name] = datasets_list
        good_list = True
    elif mode == "scalars" or mode == "vectors":
        ds[new_name] = {}
        first = True
        for dataset in datasets_list:
            if first:
                for variable in dataset.keys():
                    ds[new_name][variable] = dataset[variable].copy()
            else:
                for variable in dataset.keys():
                    if mode == "vectors":
                        out_size = len(ds[new_name][variable][0])
                        dataset_size = len(dataset[variable][0])
                        diff_size = abs(out_size-dataset_size)
                        if out_size > dataset_size:
                            number_of_events = len(dataset[variable])
                            for i in range(diff_size):
                                dataset[variable] = np.c_[ dataset[variable], np.zeros(number_of_events) ]
                        elif dataset_size > out_size:
                            number_of_events = len(ds[new_name][variable])
                            for i in range(diff_size):
                                ds[new_name][variable] = np.c_[ ds[new_name][variable], np.zeros(number_of_events) ]
                    ds[new_name][variable] = np.concatenate((ds[new_name][variable],dataset[variable]))
            first = False
        good_list = True

    else:
        print("Type of the items is not supported!")

    if good_list:
        for input_name in input_list:
            if input_name != new_name:
                del ds[input_name]

    del datasets_list


#==================================================================================================
class control:
    """
    Produce control information to assist in the definition of cuts
    """
    def __init__(self, var, signal_list, others_list, weight=None, bins=np.linspace(0,100,5), above=True):
        self.bins = bins
        self.var = var
        self.signal_list = signal_list
        self.others_list = others_list
        self.weight = weight

        use_bins = [np.array([-np.inf]), np.array(bins), np.array([np.inf])]
        use_bins = np.concatenate(use_bins)

        hist_signal_list = []
        for signal in signal_list:
            if weight is not None:
                hist, hbins = np.histogram( signal[var], weights=signal[weight], bins=use_bins )
            else:
                hist, hbins = np.histogram( signal[var], bins=use_bins )
            if not above:
                hist = np.cumsum(hist)
                hist = hist[:-1]
            else:
                hist = np.cumsum(hist[::-1])[::-1]
                hist = hist[1:]
            hist_signal_list.append(hist)
        hist_signal = hist_signal_list[0]
        for i in range(len(signal_list)-1):
            hist_signal = hist_signal + hist_signal_list[i+1]
        self.hist_signal = hist_signal

        hist_others_list = []
        for others in others_list:
            if weight is not None:
                hist, hbins = np.histogram( others[var], weights=others[weight], bins=use_bins )
            else:
                hist, hbins = np.histogram( others[var], bins=use_bins )
            if not above:
                hist = np.cumsum(hist)
                hist = hist[:-1]
            else:
                hist = np.cumsum(hist[::-1])[::-1]
                hist = hist[1:]
            hist_others_list.append(hist)
        hist_others = hist_others_list[0]
        for i in range(len(others_list)-1):
            hist_others = hist_others + hist_others_list[i+1]
        self.hist_others = hist_others

        signal_sum_list = []
        for signal in signal_list:
            if weight is not None:
                signal_sum = signal[weight].sum()
            else:
                signal_sum = len(signal[var])
            signal_sum_list.append(signal_sum)
        full_signal = signal_sum_list[0]
        for i in range(len(signal_list)-1):
            full_signal = full_signal + signal_sum_list[i+1]
        self.full_signal = full_signal

        others_sum_list = []
        for others in others_list:
            if weight is not None:
                others_sum = others[weight].sum()
            else:
                others_sum = len(others[var])
            others_sum_list.append(others_sum)
        full_others = others_sum_list[0]
        for i in range(len(others_list)-1):
            full_others = full_others + others_sum_list[i+1]
        self.full_others = full_others

        self.purity = self.hist_signal/(self.hist_signal + self.hist_others)
        self.eff_signal = self.hist_signal/self.full_signal
        self.eff_others = self.hist_others/self.full_others
        self.rej_others = 1 - self.eff_others
        self.ams = self.eff_signal/np.sqrt(self.eff_signal + self.eff_others + 1.E-7)

    #--------------------------------------------------------------------------------------
    def roc_plot(self, label='Signal-bkg ROC', color='blue', linestyle="-", version=1):
        if version == 1:
            plt.plot(self.rej_others, self.eff_signal, color=color, label=label, linestyle=linestyle)
        elif version == 2:
            plt.plot(self.eff_signal, self.eff_others, color=color, label=label, linestyle=linestyle)

    #--------------------------------------------------------------------------------------
    def auc(self, method="trapezoidal"):
        if method == "sklearn":
            area = metrics.auc(self.rej_others, self.eff_signal)
        if method == "trapezoidal":
            area = 0
            for i in range(len(self.bins)-1):
                area += 0.5*(self.eff_signal[i+1] + self.eff_signal[i])*abs(self.rej_others[i+1] - self.rej_others[i])
        return area  
            
    #--------------------------------------------------------------------------------------
    def ams_plot(self, label='Signal-bkg AMS', color='blue', linestyle="-"):
        plt.plot(self.rej_others, self.ams, color=color, label=label, linestyle=linestyle)

    #--------------------------------------------------------------------------------------
    def ams_max(self):
        return np.max(self.ams)


#===================================================================================================
def features_pca( ds_full_train_pca, variables, var_names, var_use, stat_values, class_names_pca, class_labels_pca, class_colors_pca, plots_outpath ):

    ds_full_train_pca = pd.DataFrame.from_dict(ds_full_train_pca)
    w = np.array(ds_full_train_pca['mvaWeight']).ravel()
    classes = np.array(ds_full_train_pca['class']).ravel()

    features = [ variables[i] for i in range(len(variables)) if var_use[i] == "F"]
    data_x = ds_full_train_pca[features]
    x = data_x.to_numpy()
    x_mean = np.array([ stat_values["mean"][i] for i in range(len(variables)) if var_use[i] == "F"])
    x_std = np.array([ stat_values["std"][i] for i in range(len(variables)) if var_use[i] == "F"])
    x = (x - x_mean) / x_std

    w_cov = w.copy()
    w_cov[w_cov < 0] = 0
    covariance_matrix = np.cov(x, rowvar=False, aweights=w_cov, ddof=0)
    eigenvalues_pca, eigenvectors_pca = np.linalg.eig(covariance_matrix)
    x_pca = np.matmul(x, eigenvectors_pca)

    mean_pca = []
    std_pca = []
    for i in range(len(features)):
        weighted_stats = DescrStatsW(x_pca[:,i], weights=w, ddof=0)
        mean_pca.append(weighted_stats.mean)
        std_pca.append(weighted_stats.std)
    np.set_printoptions(legacy='1.21')
    print("mean_pca: " + str(mean_pca))
    print("std_pca: " + str(std_pca))
    print("eigenvalues_pca: " + str(eigenvalues_pca))

    n_features = len(features)
    n_param = len(variables)-len(features)
    eigenvectors_ext_pca = np.block([
        [eigenvectors_pca,                np.zeros((n_features, n_param))],
        [np.zeros((n_param, n_features)), np.eye(n_param)                ]
    ])
    mean_ext_pca = np.block([np.array(mean_pca), np.zeros(n_param)])
    std_ext_pca = np.block([np.array(std_pca), np.ones(n_param)])

    pca_values={"mean_pca": mean_ext_pca, "std_pca": std_ext_pca, "eigenvectors_pca": eigenvectors_ext_pca}

    #------------------------------------------------------------------------------------
    # Emulate format of variables in the NN input
    """
    data_x_ext = ds_full_train_pca[variables]
    x_ext = data_x_ext.to_numpy()
    x_ext_mean = np.array(stat_values["mean"])
    x_ext_std = np.array(stat_values["std"])
    x_ext = (x_ext - x_ext_mean) / x_ext_std
    x_ext_pca = np.matmul(x_ext, eigenvectors_ext_pca)
    """
    #------------------------------------------------------------------------------------

    col_names = ["PCA_component_"+str(i) for i in np.arange(x_pca.shape[1])]
    df_pca = pd.DataFrame(x_pca, columns=col_names)
    df_pca['mvaWeight'] = w
    df_pca['class'] = classes


    for i in range(len(features)):
        fig1 = plt.figure(figsize=(9,5))
        gs1 = gs.GridSpec(1, 1)
        #==================================================
        ax1 = plt.subplot(gs1[0])
        #==================================================
        var = col_names[i]
        bins = np.linspace(mean_pca[i]-5*std_pca[i],mean_pca[i]+5*std_pca[i],101)
        for ikey in range(len(class_names_pca)):
            tools.step_plot( ax1, var, df_pca[df_pca["class"] == ikey], label=class_labels_pca[ikey]+" (train)", color=class_colors_pca[ikey], weight="mvaWeight", bins=bins, error=True )


        #tools.step_plot( ax1, var, df_pca, label="Train sample", color="blue", weight="mvaWeight", bins=bins, error=True )
        ax1.set_xlabel(col_names[i], size=14, horizontalalignment='right', x=1.0)
        ax1.set_ylabel("Events normalized", size=14, horizontalalignment='right', y=1.0)

        ax1.tick_params(which='major', length=8)
        ax1.tick_params(which='minor', length=4)
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.spines['bottom'].set_linewidth(1)
        ax1.spines['top'].set_linewidth(1)
        ax1.spines['left'].set_linewidth(1)
        ax1.spines['right'].set_linewidth(1)
        ax1.margins(x=0)
        ax1.legend(numpoints=1, ncol=2, prop={'size': 10.5}, frameon=False)

        plt.subplots_adjust(left=0.09, bottom=0.115, right=0.97, top=0.95, wspace=0.18, hspace=0.165)
        plt.savefig(os.path.join(plots_outpath, var + '.png'), dpi=400)
        plt.savefig(os.path.join(plots_outpath, var + '.pdf'))
        plt.close()

    return pca_values
    

#===================================================================================================
def step_plot( ax, var, dataframe, label, color='black', weight=None, error=False, normalize=False, bins=np.linspace(0,100,5), linestyle='solid', overflow=False, underflow=False ):


    if weight is None:
        W = None
        W2 = None
    else:
        W = dataframe[weight]
        W2 = dataframe[weight]*dataframe[weight]

    eff_bins = bins[:]
    if overflow:
        eff_bins[-1] = np.inf
    if underflow:
        eff_bins[0] = -np.inf

    counts, binsW = np.histogram(
        dataframe[var],
        bins=eff_bins,
        weights=W
    )
    yMC = np.array(counts)

    countsW2, binsW2 = np.histogram(
        dataframe[var],
        bins=eff_bins,
        weights=W2
    )
    errMC = np.sqrt(np.array(countsW2))

    if normalize:
        if weight is None:
            norm_factor = len(dataframe[var])
        else:
            norm_factor = dataframe[weight].sum()
        yMC = yMC/norm_factor
        errMC = errMC/norm_factor

    ext_yMC = np.append([yMC[0]], yMC)

    plt.step(bins, ext_yMC, color=color, label=label, linewidth=1.5, linestyle=linestyle)

    if error:
        x = np.array(bins)
        dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
        x = x[:-1]

        ax.errorbar(
            x+0.5*dx,
            yMC,
            yerr=[errMC, errMC],
            fmt=',',
            color=color,
            elinewidth=1
        )

    return yMC, errMC


#==================================================================================================
def confusion_matrix_plot(ax, y_true, y_pred, weights, classes, normalize='row', cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    
    if normalize == 'row':
        title = 'Confusion matrix - row normalized'
    if normalize == 'column':
        title = 'Confusion matrix - column normalized'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, sample_weight=weights)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    
    if normalize == 'row':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if normalize == 'column':
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    
    # Plot normalized confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    #ax.figure.colorbar(im, ax=ax, pad=0)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title=title, ylabel='True class', xlabel='Predicted class')
    plt.minorticks_off()

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.xlim(-0.5, len(np.unique(y_true))-0.5)
    plt.ylim(len(np.unique(y_true))-0.5, -0.5)

    np.set_printoptions(precision=2)

    return cm


#==================================================================================================
# Define a function to plot model parameters in pytorch
def print_model_parameters(model):
    count = 0
    for ele in model.state_dict():
        count += 1
        if count % 2 != 0:
            print ("The following are the parameters for the layer ", count // 2 + 1)
        if ele.find("bias") != -1:
            print("The size of bias: ", model.state_dict()[ele].size())
        else:
            print("The size of weights: ", model.state_dict()[ele].size())


#==================================================================================================
# Torch losses
class MSE_loss(nn.Module): # use it for regression
    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, y_true, y_pred, weight, device="cpu"):

        epsilon = 1e-7
        y_pred = (1-2*epsilon)*y_pred + epsilon


        total_mse_loss = torch.sum(((y_pred - y_true)**2)*torch.abs(weight))
        num_of_samples = torch.sum(torch.abs(weight))
        mean_mse_loss = total_mse_loss / num_of_samples

        return mean_mse_loss


class BCE_loss(nn.Module): # use with sigmoid
    def __init__(self):
        super(BCE_loss, self).__init__()

    def forward(self, y_true, y_pred, weight, device="cpu"):

        epsilon = 1e-7
        y_pred = (1-2*epsilon)*y_pred + epsilon


        total_bce_loss = torch.sum((-y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred))*torch.abs(weight))
        num_of_samples = torch.sum(torch.abs(weight))
        mean_bce_loss = total_bce_loss / num_of_samples

        return mean_bce_loss


class CCE_loss(nn.Module): # use with softmax
    def __init__(self, num_classes):
        super(CCE_loss, self).__init__()

        self.num_classes = num_classes

    def forward(self, y_true, y_pred, weight, device="cpu"):

        epsilon = 1e-7
        y_pred = (1-2*epsilon)*y_pred + epsilon

        if device == "cuda":
            y_true = torch.eye(self.num_classes, device="cuda")[y_true[:,0]]
        else:
            y_true = torch.eye(self.num_classes)[y_true[:,0]]

        loss_n = -torch.sum(y_true*torch.log(y_pred), dim=-1).view(-1,1)

        total_ce_loss = torch.sum(loss_n*torch.abs(weight))
        num_of_samples = torch.sum(torch.abs(weight))
        mean_ce_loss = total_ce_loss / num_of_samples

        return mean_ce_loss



#==================================================================================================
def batch_generator(data, batch_size):
    #Generate batches of data.

    #Given a list of numpy data, it iterates over the list and returns batches of the same size
    #This
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = numpy_random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr



#==================================================================================================
def train_model(input_path, N_signal, train_frac, load_size, parameters, variables, var_names, var_use, classes, reweight_info, domains=None, n_iterations=5000, mode="torch", stat_values=None, eval_step_size=0.2, eval_interval=1, feature_info=False, vec_variables=[], vec_var_names=[], vec_var_use=[], early_stopping=300, device="cpu", initial_model_path=None, plots_outpath=None):

    n_classes = len(classes)
    period = parameters[1]

    model_type = parameters[0]
    batch_size = parameters[5]

    if mode == "torch":

        iteration_cum = 0
        iteration = []
        train_acc = []
        test_acc = []
        train_loss = []
        test_loss = []
        domains_train_acc = []
        domains_test_acc = []
        domains_train_loss = []
        domains_test_loss = []
        position = 0
        min_loss = 99999
        
        torch.set_num_threads(6)

        # Criterion
        if parameters[4] == 'cce':
            criterion = CCE_loss(num_classes=n_classes)
        elif parameters[4] == 'bce':
            criterion = BCE_loss()
        elif parameters[4] == 'mse':
            criterion = MSE_loss()

        # Model
        full_model = build_model(model_type, parameters, n_classes, stat_values, variables, var_use, vec_variables, vec_var_use, device)
        if device == "cuda":
            full_model = nn.DataParallel(full_model) # Wrap the model with DataParallel
            full_model = full_model.to('cuda') # Move the model to the GPU
        #print(list(full_model.parameters()))
        print(" ")
        print(full_model.parameters)
        print(" ")

        #checkpoint_path='checkpoint_model.pt'
        checkpoint={'iteration':None, 'model_state_dict':None, 'optimizer_state_dict':None, 'loss': None}

        if "DANN" in model_type:
            encoder = build_encoder(parameters, variables, stat_values, device)
            classifier = build_classifier(parameters, n_classes)
            discriminators = []
            for idi in range(len(domains)):
                discriminators.append(build_discriminator(parameters, len(domains[idi])))
            if device == "cuda":
                encoder = nn.DataParallel(encoder)
                encoder = encoder.to('cuda')
                classifier = nn.DataParallel(classifier)
                classifier = classifier.to('cuda')
                for idi in range(len(domains)):
                    discriminators[idi] = nn.DataParallel(discriminators[idi])
                    discriminators[idi] = discriminators[idi].to('cuda')


        if initial_model_path is not None: # not valid for DANN
            full_model.load_state_dict(torch.load(initial_model_path, weights_only=True))

        early_stopping_count = 0
        load_it = 0
        period_count = 0
        waiting_period = 99999
        verbose = True
        for i in tqdm(range(n_iterations)):

            p = i/n_iterations
            alpha = 1 #2. / (1. + np.exp(-10 * p)) - 1

            if ((load_it == 0) or (period_count == waiting_period)) and (iteration_cum == 0):
                ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, class_colors, reweight_info = get_sample(input_path, period, classes, N_signal, train_frac, load_size, load_it, reweight_info, features=variables+["evtWeight"], vec_features=vec_variables, verbose=verbose)

                train_data = process_data(model_type, ds_full_train, vec_full_train, variables, vec_variables, var_use, vec_var_use, stat_values, device, parameters)
                test_data = process_data(model_type, ds_full_test, vec_full_test, variables, vec_variables, var_use, vec_var_use, stat_values, device, parameters)

                #np.set_printoptions(legacy='1.21')
                train_batches = batch_generator(train_data, batch_size)

                #=======TEST PLOTS========================================
                #tools.features_stat(model_type, ds_full_train, ds_full_test, vec_full_train, vec_full_test, variables, vec_variables, var_names, vec_var_names, var_use, vec_var_use, class_names, class_labels, class_colors, plots_outpath, load_it=load_it)
                #=======TEST PLOTS========================================

                waiting_period = int(len(ds_full_train['mvaWeight'])/batch_size)
                del ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, class_colors

                domain_train_data = []
                domain_test_data = []
                if "DANN" in model_type:
                    domain_train_batches = []
                    for idi in range(len(domains)):
                        ds_domain_full_train, ds_domain_full_test, vec_domain_full_train, vec_domain_full_test, domain_names, domain_labels, domain_colors, _ = get_sample(input_path, period, domains[idi], N_signal, train_frac, load_size, load_it, reweight_info, features=variables+["evtWeight"], vec_features=vec_variables, verbose=verbose)

                        domain_train_data.append(process_data(model_type, ds_domain_full_train, vec_domain_full_train, variables, vec_variables, var_use, vec_var_use, stat_values, device, parameters))
                        domain_test_data.append(process_data(model_type, ds_domain_full_test, vec_domain_full_test, variables, vec_variables, var_use, vec_var_use, stat_values, device, parameters))

                        domain_train_batches.append(batch_generator(domain_train_data[idi], batch_size))

                        #=======TEST PLOTS========================================
                        #tools.features_stat(model_type, ds_domain_full_train, ds_domain_full_test, vec_domain_full_train, vec_domain_full_test, variables, vec_variables, var_names, vec_var_names, var_use, vec_var_use, domain_names, domain_labels, domain_colors, plots_outpath, load_it=load_it)
                        #=======TEST PLOTS========================================

                        del ds_domain_full_train, ds_domain_full_test, vec_domain_full_train, vec_domain_full_test, domain_names, domain_labels, domain_colors

                if verbose:
                    verbose = False
                load_it += 1
                period_count = 0




            batch_data = next(train_batches)
            train_w_sum = train_data[-1].sum()
            test_w_sum = test_data[-1].sum()
            domain_batch_data = []
            domain_train_w_sum = []
            domain_test_w_sum = []
            if "DANN" not in model_type:
                full_model.train()
                full_model = update_model(model_type, full_model, criterion, parameters, batch_data, domain_batch_data, alpha, stat_values, var_use, device)
                full_model.eval()
            else:
                for idi in range(len(domains)):
                    domain_batch_data.append(next(domain_train_batches[idi]))
                    domain_train_w_sum.append(domain_train_data[idi][-1].sum())
                    domain_test_w_sum.append(domain_test_data[idi][-1].sum())

                encoder.train()
                classifier.train()
                for idi in range(len(domains)):
                    discriminators[idi].train()
                training_model = [encoder, classifier, discriminators]
                encoder, classifier, discriminators = update_model(model_type, training_model, criterion, parameters, batch_data, domain_batch_data, alpha, stat_values, var_use, device)
                encoder.eval()
                classifier.eval()
                for idi in range(len(domains)):
                    discriminators[idi].eval()
                full_model.encoder.load_state_dict(encoder.encoder.state_dict())
                full_model.classifier.load_state_dict(classifier.classifier.state_dict())


            n_eval_train_steps = int(len(train_data[-1])/eval_step_size) + 1
            n_eval_test_steps = int(len(test_data[-1])/eval_step_size) + 1


            period_count += 1

            #------------------------------------------------------------------------------------
            if ((i + 1) % eval_interval == 0):

                with torch.no_grad():
                    if "DANN" in model_type:
                        evaluation_model = [encoder, classifier, discriminators]
                    else:
                        evaluation_model = full_model

                    train_loss_i = 0
                    train_acc_i = 0
                    domains_train_loss_i = [0 for i in range(len(domains))]
                    domains_train_acc_i = [0 for i in range(len(domains))]
                    for i_eval in range(n_eval_train_steps):
                        i_eval_output = evaluate_model(model_type, train_data, evaluation_model, i_eval, eval_step_size, criterion, parameters, stat_values, var_use, domain_train_data, alpha, device, mode="metric")
                        if i_eval_output is None:
                            continue
                        else:
                            if "DANN" in model_type:
                                i_eval_loss, i_eval_acc, i_eval_domains_loss, i_eval_domains_acc = i_eval_output
                            else:
                                i_eval_loss, i_eval_acc = i_eval_output
                        train_loss_i += i_eval_loss
                        train_acc_i += i_eval_acc
                        if "DANN" in model_type:
                            for idi in range(len(domains)):
                                domains_train_loss_i[idi] += i_eval_domains_loss[idi]
                                domains_train_acc_i[idi] += i_eval_domains_acc[idi]
                    train_loss_i = train_loss_i/train_w_sum
                    train_acc_i = train_acc_i/train_w_sum
                    if "DANN" in model_type:
                        for idi in range(len(domains)):
                            domains_train_loss_i[idi] = domains_train_loss_i[idi]/domain_train_w_sum[idi]
                            domains_train_acc_i[idi] = domains_train_acc_i[idi]/domain_train_w_sum[idi]


                    test_loss_i = 0
                    test_acc_i = 0
                    domains_test_loss_i = [0 for i in range(len(domains))]
                    domains_test_acc_i = [0 for i in range(len(domains))]
                    for i_eval in range(n_eval_test_steps):
                        i_eval_output = evaluate_model(model_type, test_data, evaluation_model, i_eval, eval_step_size, criterion, parameters, stat_values, var_use, domain_train_data, alpha, device, mode="metric")
                        if i_eval_output is None:
                            continue
                        else:
                            if "DANN" in model_type:
                                i_eval_loss, i_eval_acc, i_eval_domains_loss, i_eval_domains_acc = i_eval_output
                            else:
                                i_eval_loss, i_eval_acc = i_eval_output
                        test_loss_i += i_eval_loss
                        test_acc_i += i_eval_acc
                        if "DANN" in model_type:
                            for idi in range(len(domains)):
                                domains_test_loss_i[idi] += i_eval_domains_loss[idi]
                                domains_test_acc_i[idi] += i_eval_domains_acc[idi]
                    test_loss_i = test_loss_i/test_w_sum
                    test_acc_i = test_acc_i/test_w_sum
                    if "DANN" in model_type:
                        for idi in range(len(domains)):
                            domains_test_loss_i[idi] = domains_test_loss_i[idi]/domain_test_w_sum[idi]
                            domains_test_acc_i[idi] = domains_test_acc_i[idi]/domain_test_w_sum[idi]

                    del evaluation_model

                    iteration.append(iteration_cum+i+1)
                    train_acc.append(train_acc_i)
                    test_acc.append(test_acc_i)
                    train_loss.append(train_loss_i)
                    test_loss.append(test_loss_i)
                    domains_train_acc.append(domains_train_acc_i)
                    domains_test_acc.append(domains_test_acc_i)
                    domains_train_loss.append(domains_train_loss_i)
                    domains_test_loss.append(domains_test_loss_i)

                    if( (test_loss_i < min_loss) ):
                        min_loss = test_loss_i
                        position = i+1
                        #checkpoint['iteration']=iteration
                        checkpoint['model_state_dict']=full_model.state_dict()
                        #checkpoint['optimizer_state_dict']= optimizer.state_dict()
                        checkpoint['loss']=min_loss
                        early_stopping_count = 0
                    else:
                        early_stopping_count += 1

                    print("iterations %d, class loss =  %.10f, class accuracy =  %.3f"%(i+1, test_loss_i, test_acc_i ))
                    if "DANN" in model_type:
                        for idi in range(len(domains)):
                            print("DI %d, domain loss =  %.10f, domain accuracy =  %.3f"%(idi, domains_test_loss_i[idi], domains_test_acc_i[idi] ))

                    if early_stopping_count == early_stopping:
                        print("Early stopping activated!")
                        break

        #--------------------------------------------------------------------------------------
        if( position > 0 ):
            # Set weights of the best classification model
            #print(checkpoint['model_state_dict'])
            if early_stopping is not None:
                full_model.load_state_dict(checkpoint['model_state_dict'])
            min_loss = checkpoint['loss']

            # Permutation feature importance
            # https://cms-ml.github.io/documentation/optimization/importance.html
            feature_score_info = []
            if feature_info:
                print("")
                print("------------------------------------------------------------------------")
                print("Computing feature importance")
                print("------------------------------------------------------------------------")
                with torch.no_grad():
                    feature_score_info = feature_score(model_type, test_data, full_model, min_loss, eval_step_size, criterion, parameters, variables, vec_variables, var_use, vec_var_use, var_names, vec_var_names, stat_values, device)

    return full_model, np.array(iteration), np.array(train_acc), np.array(test_acc), np.array(train_loss), np.array(test_loss), np.array(domains_train_acc), np.array(domains_test_acc), np.array(domains_train_loss), np.array(domains_test_loss), feature_score_info



#==================================================================================================
def evaluate_models(period, library, tag, outpath_base, modelNames_submitted, models_submitted, condor):

    models_submitted = [model[3:] for model in models_submitted]
    models_dict = dict(zip(modelNames_submitted, models_submitted))
    
    best_models_path = os.path.join(outpath_base, period, "best_models")
    if not os.path.exists(best_models_path):
        os.makedirs(best_models_path)
    
    if condor:
        storage_user = os.environ.get("STORAGE_USER")
        outpath_info = outpath_base.split("/")
        outpath_base = os.path.join("/cms/store/user/", storage_user, outpath_info[-4], outpath_info[-3], outpath_info[-2], outpath_info[-1])

    list_signals = os.listdir(os.path.join(outpath_base, period, library, tag))
    if 'best_models.csv' in list_signals:
        list_signals.remove('best_models.csv')

    print("#########################################################################################")
    print(library)
    print("#########################################################################################")

    ml_outpath = os.path.join(outpath_base, period, library, tag)
    os.system("rm -rf " + os.path.join(best_models_path, library, tag))
    print("outpath = ", ml_outpath)

    list_best_models = []
    for signal in list_signals:
        #print(signal)
        list_models = os.listdir(os.path.join(ml_outpath, signal, "models"))
        models_loss = []
        models_accuracy = []
        models_iterations = []
        models_name = []
        models_hyperparameters = []
        for model in list_models:
            #print(model)
            training_file = os.path.join(ml_outpath, signal, "models", model, "training.csv")
            if os.path.isfile(training_file):
                df_training = pd.read_csv(training_file)
                if len(df_training) > 0 and len(df_training["test_loss"]) > 0 and len(df_training["test_acc"]) > 0:
                    min_loss = np.amin(df_training["test_loss"])
                    if not math.isnan(min_loss):
                        models_loss.append(min_loss)
                        models_accuracy.append(np.array(df_training[df_training["test_loss"] == min_loss]["test_acc"])[-1])
                        models_iterations.append(np.array(df_training[df_training["test_loss"] == min_loss]["iteration"])[-1])
                        models_name.append(model)
                        models_hyperparameters.append(models_dict[model])
        df_training = pd.DataFrame({"Model": models_name, "Loss": models_loss, "Accuracy": models_accuracy, "Iterations": models_iterations, "Hyperparameters": models_hyperparameters})
        df_training = df_training.sort_values("Loss")
        df_training = df_training.reset_index(drop=True)

        #best_model_dir = os.path.join(ml_outpath, signal, "models", df_training.loc[0]["Model"])
        #signal_dir = os.path.join(ml_outpath, signal)
        #copyCommand = "cp -rf " + best_model_dir + " " + signal_dir
        #os.system(copyCommand)
        #print(signal)
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXX")
        #print(df_training)

        list_best_models.append(df_training.loc[0]["Model"])

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option("display.precision", 6)
        print("====================================================================================")
        print(signal)
        print("====================================================================================")
        print(df_training)
        print("")
        save_path = os.path.join(best_models_path, library, tag, signal)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_training.to_csv(os.path.join(save_path, "training_result.csv"))

        os.system("cp -rf " + os.path.join(ml_outpath, signal, 'features') + " " + save_path)

        models_path = os.path.join(save_path, 'models')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        if len(df_training) <= 1:
            for model in list_models:
                copyCommand = "cp -rf " + os.path.join(ml_outpath, signal, 'models', model) + " " + models_path
                os.system(copyCommand)
        else:
            for model in list_models:
                #if( model == df_training.loc[0]["Model"] or model == df_training.loc[1]["Model"] or model == df_training.loc[2]["Model"] ):
                if( model == df_training.loc[0]["Model"] ):
                    copyCommand = "cp -rf " + os.path.join(ml_outpath, signal, 'models', model) + " " + models_path
                    os.system(copyCommand)



    df_result = pd.DataFrame({"signal": list_signals, "best model": list_best_models})
    df_result = df_result.reset_index()
    df_result.to_csv(os.path.join(best_models_path, library, tag, "best_models.csv"))










#==================================================================================================
def build_model(model_type, parameters, n_classes, stat_values, variables, var_use, vec_variables, vec_var_use, device):

    if model_type == "NN":
        model = build_NN(parameters, variables, n_classes, stat_values, device)
    elif model_type == "PNN":
        model = build_PNN(parameters, variables, n_classes, stat_values, device)
    elif model_type == "APNN":
        model = build_APNN(parameters, variables, n_classes, stat_values, device)
    elif model_type == "APSNN":
        model = build_APSNN(parameters, variables, n_classes, stat_values, device)
    elif model_type == "PNET":
        model = build_PNET(vec_variables, vec_var_use, n_classes, parameters, stat_values, device)
    elif model_type == "M2CNN":
        model = build_M2CNN(vec_variables, n_classes, parameters, stat_values, device)
    elif model_type == "DANN":
        model = build_DANN(parameters, variables, n_classes, stat_values, device)

    return model


#==================================================================================================
def model_parameters(model_type, param_dict):

    if model_type == "NN":
        model_parameters = model_parameters_NN(param_dict)
    elif model_type == "PNN":
        model_parameters = model_parameters_PNN(param_dict)
    elif model_type == "APNN":
        model_parameters = model_parameters_APNN(param_dict)
    elif model_type == "APSNN":
        model_parameters = model_parameters_APSNN(param_dict)
    elif model_type == "PNET":
        model_parameters = model_parameters_PNET(param_dict)
    elif model_type == "M2CNN":
        model_parameters = model_parameters_M2CNN(param_dict)
    elif model_type == "DANN":
        model_parameters = model_parameters_DANN(param_dict)

    return model_parameters


#==================================================================================================
def features_stat(model_type, train_data, test_data, vec_train_data, vec_test_data, variables, vec_variables, var_names, vec_var_names, var_use, vec_var_use, class_names, class_labels, class_colors, plots_outpath, load_it=None, parameters=None):

    if model_type == "NN":
        stat_values = features_stat_NN(train_data, test_data, variables, var_names, class_names, class_labels, class_colors, plots_outpath, load_it=load_it)
    elif model_type == "PNN":
        stat_values = features_stat_PNN(train_data, test_data, variables, var_names, var_use, class_names, class_labels, class_colors, plots_outpath, load_it=load_it)
    elif model_type == "APNN":
        stat_values = features_stat_APNN(train_data, test_data, variables, var_names, var_use, class_names, class_labels, class_colors, plots_outpath, load_it=load_it)
    elif model_type == "APSNN":
        stat_values = features_stat_APSNN(train_data, test_data, variables, var_names, var_use, class_names, class_labels, class_colors, plots_outpath, parameters, load_it=load_it)
    elif model_type == "PNET":
        stat_values = features_stat_PNET(train_data, test_data, vec_train_data, vec_test_data, vec_variables, vec_var_names, vec_var_use, class_names, class_labels, class_colors, plots_outpath, load_it=load_it)
    elif model_type == "M2CNN":
        stat_values = features_stat_M2CNN(train_data, test_data, vec_train_data, vec_test_data, vec_variables, vec_var_names, class_names, class_labels, class_colors, plots_outpath, parameters, load_it=load_it)
    elif model_type == "DANN":
        stat_values = features_stat_DANN(train_data, test_data, variables, var_names, class_names, class_labels, class_colors, plots_outpath, load_it=load_it)

    return stat_values


#==================================================================================================
def update_model(model_type, model, criterion, parameters, batch_data, domain_batch_data, alpha, stat_values, var_use, device):

    if model_type == "NN":
        model = update_NN(model, criterion, parameters, batch_data, device)
    elif model_type == "PNN":
        model = update_PNN(model, criterion, parameters, batch_data, stat_values, var_use, device)
    elif model_type == "APNN":
        model = update_APNN(model, criterion, parameters, batch_data, stat_values, var_use, device)
    elif model_type == "APSNN":
        model = update_APSNN(model, criterion, parameters, batch_data, var_use, device)
    elif model_type == "PNET":
        model = update_PNET(model, criterion, parameters, batch_data, device)
    elif model_type == "M2CNN":
        model = update_M2CNN(model, criterion, parameters, batch_data, device)
    elif model_type == "DANN":
        model = update_DANN(model, criterion, parameters, batch_data, domain_batch_data, alpha, device)

    return model


#==================================================================================================
def process_data(model_type, scalar_var, vector_var, variables, vec_variables, var_use, vec_var_use, stat_values, device, parameters):

    if model_type == "NN":
        input_data = process_data_NN(scalar_var, variables)
    elif model_type == "PNN":
        input_data = process_data_PNN(scalar_var, variables)
    elif model_type == "APNN":
        input_data = process_data_APNN(scalar_var, variables)
    elif model_type == "APSNN":
        input_data = process_data_APSNN(scalar_var, variables, var_use, vector_var, stat_values, device)
    elif model_type == "PNET":
        input_data = process_data_PNET(scalar_var, vector_var, vec_variables, vec_var_use)
    elif model_type == "M2CNN":
        input_data = process_data_M2CNN(scalar_var, vector_var, vec_variables, parameters)
    elif model_type == "DANN":
        input_data = process_data_DANN(scalar_var, variables)

    return input_data


#==================================================================================================
def evaluate_model(model_type, input_data, model, i_eval, eval_step_size, criterion, parameters, stat_values, var_use, domain_input_data, alpha, device, mode="predict"):

    if model_type == "NN":
        i_eval_output = evaluate_NN(input_data, model, i_eval, eval_step_size, criterion, parameters, device, mode)
    elif model_type == "PNN":
        i_eval_output = evaluate_PNN(input_data, model, i_eval, eval_step_size, criterion, parameters, stat_values, var_use, device, mode)
    elif model_type == "APNN":
        i_eval_output = evaluate_APNN(input_data, model, i_eval, eval_step_size, criterion, parameters, stat_values, var_use, device, mode)
    elif model_type == "APSNN":
        i_eval_output = evaluate_APSNN(input_data, model, i_eval, eval_step_size, criterion, parameters, stat_values, var_use, device, mode)
    elif model_type == "PNET":
        i_eval_output = evaluate_PNET(input_data, model, i_eval, eval_step_size, criterion, parameters, device, mode)
    elif model_type == "M2CNN":
        i_eval_output = evaluate_M2CNN(input_data, model, i_eval, eval_step_size, criterion, parameters, device, mode)
    elif model_type == "DANN":
        i_eval_output = evaluate_DANN(input_data, model, i_eval, eval_step_size, criterion, parameters, domain_input_data, alpha, device, mode)

    return i_eval_output


#==================================================================================================
def feature_score(model_type, input_data, model, min_loss, eval_step_size, criterion, parameters, variables, vec_variables, var_use, vec_var_use, var_names, vec_var_names, stat_values, device):

    if model_type == "NN":
        feature_score_info = feature_score_NN(input_data, model, min_loss, eval_step_size, criterion, parameters, variables, var_names, device)
    elif model_type == "PNN":
        feature_score_info = feature_score_PNN(input_data, model, min_loss, eval_step_size, criterion, parameters, variables, var_names, var_use, stat_values, device)
    elif model_type == "APNN":
        feature_score_info = feature_score_APNN(input_data, model, min_loss, eval_step_size, criterion, parameters, variables, var_names, var_use, stat_values, device)
    elif model_type == "APSNN":
        feature_score_info = feature_score_APSNN(input_data, model, min_loss, eval_step_size, criterion, parameters, variables, var_names, var_use, stat_values, device)
    elif model_type == "PNET":
        feature_score_info = feature_score_PNET(input_data, model, min_loss, eval_step_size, criterion, parameters, vec_variables, vec_var_use, vec_var_names, device)
    elif model_type == "M2CNN":
        feature_score_info = feature_score_M2CNN(input_data, model, min_loss, eval_step_size, criterion, parameters, vec_variables, vec_var_names, device)
    elif model_type == "DANN":
        feature_score_info = feature_score_DANN(input_data, model, min_loss, eval_step_size, criterion, parameters, variables, var_names, device)

    return feature_score_info


#==================================================================================================
def save_model(model_type, model, model_outpath, dim, device):

    if model_type == "NN":
        save_NN(model, model_outpath, dim, device)
    elif model_type == "PNN":
        save_PNN(model, model_outpath, dim, device)
    elif model_type == "APNN":
        save_APNN(model, model_outpath, dim, device)
    elif model_type == "APSNN":
        save_APSNN(model, model_outpath, dim, device)
    elif model_type == "PNET":
        save_PNET(model, model_outpath, dim, device)
    elif model_type == "M2CNN":
        save_M2CNN(model, model_outpath, dim, device)
    elif model_type == "DANN":
        save_DANN(model, model_outpath, dim, device)


