import sys
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import concurrent.futures as cf
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.ticker import AutoMinorLocator
import json
import tools
import math

#-------------------------------------------------------------------------------------
# General Setup
#-------------------------------------------------------------------------------------
input_path = '/cms/store/user/gcorreia/output/AP_bbZDM_Lep_R2/ML/datasets'
output_path = '/home/gcorreia/output/AP_bbZDM_Lep_R2/ML/'
periods = ['0_17']
tag = 'bbZDM'

#--------------------------------------------------------------------------------------------------
# ML setup
#--------------------------------------------------------------------------------------------------
device = 'cuda' # 'cuda'
library = 'torch'
optimizer = ['adam']
loss_func = ['bce']
learning_rate = [[0.01]]


#--------------------------------------------------------------------------------------------------
# Models setup
#--------------------------------------------------------------------------------------------------
model_type = 'NN'
model_parameters = {
    'num_layers': [2, 3],
    'num_nodes': [20, 30, 50],
    'activation_func': ['elu']
    }

#-------------------------------------------------------------------------------------
# Training setup
#-------------------------------------------------------------------------------------
batch_size = [10000]
load_size_stat = 1000000000
load_size_training = 1000000000
num_load_for_check = 1
train_frac = 0.5
eval_step_size = 10000
num_max_iterations = 10000
early_stopping = 20
initial_model_path = None

#-------------------------------------------------------------------------------------
# Inputs setup
#-------------------------------------------------------------------------------------
feature_info = False

scalar_variables = [
    #['jet_pt', 'Jet_pt', 'F'], 
    #['jet_mass', 'Jet_mass', 'F']
    ["LeadingLep_pt",           r"$\mathrm{leading}\,p_\mathrm{T}^\mathrm{l}$",  "F"],
    ["TrailingLep_pt",          r"$\mathrm{trailing}\,p_\mathrm{T}^\mathrm{l}$", "F"],
    ["LepLep_pt",               r"$p_\mathrm{T}^\mathrm{ll}$",                   "F"],
    ["LepLep_deltaR",           r"$\Delta R^\mathrm{ll}$",                       "F"],
    ["LepLep_deltaM",           r"$\Delta M^\mathrm{ll}$",                       "F"],
    ["MET_pt",                  r"$E_\mathrm{T}^\mathrm{miss}$",                 "F"],
    ["MET_LepLep_Mt",           r"$M_\mathrm{T}^\mathrm{ll,MET}$",               "F"],
    ["MET_LepLep_deltaPhi",     r"$\Delta \phi^\mathrm{ll,MET}$",                "F"],
    ["MT2LL",                   r"$M_\mathrm{T2}^\mathrm{ll}$",                  "F"],
]

vector_variables = []

#-------------------------------------------------------------------------------------
# Preprocessing setup
#-------------------------------------------------------------------------------------
reweight_variables = []

normalization_method = 'area'

#-------------------------------------------------------------------------------------
# Classes setup
#-------------------------------------------------------------------------------------
classes = {
#<class_name>: [[<list_of_processes>], <mode>, <combination>, <label>, <color>]
"Signal_samples": [[
    "Signal_1000_100",
    "Signal_1000_200",
    "Signal_1000_300",
    "Signal_1000_400",
    "Signal_1000_600",
    "Signal_1000_800",
    "Signal_400_100",
    "Signal_400_200",
    "Signal_500_100",
    "Signal_500_200",
    "Signal_500_300",
    "Signal_600_100",
    "Signal_600_200",
    "Signal_600_300",
    "Signal_600_400",
    "Signal_800_100",
    "Signal_800_200",
    "Signal_800_300",
    "Signal_800_400",
    "Signal_800_600",
    #"Signal_1400_100",
    #"Signal_1400_400",
    #"Signal_1400_600",
    #"Signal_1400_1000",
    #"Signal_1400_1200",
    #"Signal_2000_100",
    #"Signal_2000_400",
    #"Signal_2000_600",
    #"Signal_2000_1000",
    #"Signal_2000_1200",
    #"Signal_2000_1800",
    ], "normal", "equal", "Signal", "green"],
"Background": [[
    "DYJetsToLL_Pt-0To3",
    "DYJetsToLL_PtZ-3To50",
    "DYJetsToLL_PtZ-50To100",
    "DYJetsToLL_PtZ-100To250",
    "DYJetsToLL_PtZ-250To400",
    "DYJetsToLL_PtZ-400To650",
    "DYJetsToLL_PtZ-650ToInf",
    "TTTo2L2Nu",
    "TTToSemiLeptonic",
    "ST_tW_antitop",
    "ST_tW_top",
    "ST_s-channel",
    "ST_t-channel_top",
    "ST_t-channel_antitop",
    "WZTo3LNu",
    "ZZTo4L",
    "ZZTo2L2Nu",
    "WZZ",
    "WWZ",
    "ZZZ",
    "WWW",
    "TTWZ",
    "TTZZ",
    "TTWW",
    "TTWJetsToLNu",
    "TTWJetsToQQ",
    "TTZToQQ",
    "TTZToLL",
    "TTZToNuNu",
    "tZq_ll",
    "WWTo2L2Nu",
    "WZTo2Q2L",
    "ZZTo2Q2L",
    ], "normal", "evtsum", "Background", "red"],
}


#-------------------------------------------------------------------------------------
# [DO NOT TOUCH THIS PART]
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--job", type=int, default=0)
parser.add_argument("--check", dest='check_flag', action='store_true')
parser.set_defaults(check_flag=False)
parser.add_argument("--clean", dest='clean_flag', action='store_true')
parser.set_defaults(clean_flag=False)
parser.add_argument("--evaluate", dest='evaluate_flag', action='store_true')
parser.set_defaults(evaluate_flag=False)

args = parser.parse_args()


#===============================================================================
# EVALUATE MODELS
#===============================================================================
if args.evaluate_flag:
    for i_period in periods:
        print("==================================")
        print(i_period)
        print("==================================")
        print(" ")
        print(" ")
        tools.evaluate_models(i_period, library, tag, output_path)
    sys.exit()


#===============================================================================
# CHECK ARGUMENT
#===============================================================================
has_signal_list = False
N_signal_points = 1
Signal_class = None
for key in classes:
    if key.startswith("Signal"):
        Signal_class = key
        if key.startswith("Signal_samples"):
            has_signal_list = True
            N_signal_points = len(classes[key][0])
            break

model_parameters = tools.model_parameters(model_type, model_parameters)

modelName = []
model = []
i_job = 0
for i_signal in range(N_signal_points):
    for i_period in periods:
        for i_optimizer in optimizer:
            for i_loss_func in loss_func:
                for i_batch_size in batch_size:
                    for i_lr in learning_rate:
                        for i_model_param in model_parameters:
                            modelName.append(str(i_job)+"_"+model_type)
                            if N_signal_points == 1:
                                model.append([model_type] + [i_period] + [Signal_class] + [i_optimizer] + [i_loss_func] + [i_batch_size] + [i_lr] + [i_model_param])
                            else:
                                model.append([model_type] + [i_period] + [classes[Signal_class][0][i_signal]] + [i_optimizer] + [i_loss_func] + [i_batch_size] + [i_lr] + [i_model_param])
                            i_job += 1
                            #model[:][2] (Signal_class) is not used anywhere

N = int(args.job)
if N == -1:
    print("")
    sys.exit("Number of jobs: " + str(len(model)))
if N == -2:
    for i in range(len(model)):
        print(str(i)+"  "+str(model[i])+",")
    sys.exit("")
if N <= -3:
    sys.exit(">> Enter an integer >= -1")
if N >= len(model):
    sys.exit("There are only " + str(len(model)) + " models")


N_signal = int(N/(len(model)/N_signal_points))
if N_signal_points == 1:
    signal_tag = Signal_class
else:
    signal_tag = classes[Signal_class][0][N_signal]

#===============================================================================
# Output setup
#===============================================================================
if args.clean_flag:
    os.system("rm -rf " + os.path.join(output_path, model[N][1], "ML", library, tag, signal_tag))
    sys.exit()

ml_outpath = os.path.join(output_path, model[N][1], "ML")
if not os.path.exists(ml_outpath):
    os.makedirs(ml_outpath)

signal_outpath = os.path.join(ml_outpath, library, tag, signal_tag)
if not os.path.exists(signal_outpath):
    os.makedirs(signal_outpath)

plots_outpath = os.path.join(signal_outpath, "features")
if not os.path.exists(plots_outpath):
    os.makedirs(plots_outpath)

if not os.path.exists(os.path.join(signal_outpath, "models")):
    os.makedirs(os.path.join(signal_outpath, "models"))

model_outpath = os.path.join(signal_outpath, "models", modelName[int(args.job)])
if not os.path.exists(model_outpath):
    os.makedirs(model_outpath)

print('Results will be stored in ' + ml_outpath)


#===============================================================================
import torch


variables = [scalar_variables[i][0] for i in range(len(scalar_variables))]
var_names = [scalar_variables[i][1] for i in range(len(scalar_variables))]
var_use = [scalar_variables[i][2] for i in range(len(scalar_variables))]
var_bins = [scalar_variables[i][3] if len(scalar_variables[i]) == 4 else None for i in range(len(scalar_variables))]

vec_variables = [vector_variables[i][0] for i in range(len(vector_variables))]
vec_var_names = [vector_variables[i][1] for i in range(len(vector_variables))]
vec_var_use = [vector_variables[i][2] for i in range(len(vector_variables))]


#===============================================================================
# Preprocessing input data (modify and stay)
#===============================================================================
if device == "cpu":
    print("Training will run on CPU.")
elif device == "cuda":
    if torch.cuda.is_available():
        print("Training will run on GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available. Training will run on CPU.")
        device = "cpu"


print("")
print("------------------------------------------------------------------------")
print("Preprocessing input data")
print("------------------------------------------------------------------------")
seed = 16

ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, class_colors, reweight_info = tools.get_sample(input_path, model[N][1], classes, N_signal, train_frac, load_size_stat, 0, reweight_variables, features=variables+["evtWeight"], vec_features=vec_variables, verbose=True,
normalization_method=normalization_method)

signal_param = []

stat_values = tools.features_stat(model_type, ds_full_train, ds_full_test, vec_full_train, vec_full_test, variables, vec_variables, var_names, vec_var_names, var_use, vec_var_use, class_names, class_labels, class_colors, plots_outpath)

if args.check_flag:
    tools.check_scalars(ds_full_train, variables, var_names, var_use, var_bins, class_names, class_labels, class_colors, plots_outpath)
    sys.exit()

del ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, class_colors


#===============================================================================
# RUN TRAINING
#===============================================================================
print("")
print("------------------------------------------------------------------------")
print("Training")
print("------------------------------------------------------------------------")

start = time.time()


class_model, iteration, train_acc, test_acc, train_loss, test_loss, adv_source_acc, adv_target_acc, feature_score_info, lr_info = tools.train_model(
    input_path,
    N_signal,
    train_frac,
    load_size_training,
    model[N],
    variables,
    var_names,
    var_use,
    classes,
    reweight_info,
    n_iterations = num_max_iterations,
    mode = library,
    stat_values = stat_values,
    eval_step_size = eval_step_size,
    feature_info = feature_info,
    vec_variables=vec_variables,
    vec_var_names=vec_var_names,
    vec_var_use=vec_var_use,
    early_stopping=early_stopping,
    device=device,
    initial_model_path=initial_model_path
    )



if feature_info:
    #===============================================================================
    # SAVE FEATURE IMPORTANCE INFORMATION
    #===============================================================================
    features_score, features_score_unc, features_names = feature_score_info

    df_feature = pd.DataFrame(list(zip(features_score, features_score_unc, features_names)),columns=["Score", "Score_unc", "Feature_name"])
    df_feature = df_feature.sort_values("Score", ascending=False)
    df_feature.to_csv(os.path.join(model_outpath, 'features.csv'), index=False)

    nf = len(features_names)
    yf = 0.345*nf + 0.2
    #hf = 0.345*nf + 0.42
    fig1 = plt.figure(figsize=(9,yf))
    grid = [1, 1]
    gs1 = gs.GridSpec(grid[0], grid[1])

    ax1 = plt.subplot(gs1[0])
    y_pos = np.arange(nf)
    ax1.barh(y_pos, df_feature["Score"], xerr=df_feature["Score_unc"], align='center', color="lightsteelblue")
    ax1.set_ylim([-0.5,nf-0.5])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_feature["Feature_name"], size=14)
    ax1.invert_yaxis()  # labels read top-to-bottom
    ax1.set_xlabel("Feature score", size=14, horizontalalignment='right', x=1.0)

    ax1.tick_params(which='major', length=8)
    ax1.tick_params(which='minor', length=4)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(which='major', axis='x', linewidth=0.2, linestyle='-', color='0.75')
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)
    ax1.margins(x=0)
    ax1.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False, loc='lower right')


    plt.subplots_adjust(left=0.22, bottom=0.5/yf, right=0.97, top=0.99, wspace=0.18, hspace=0.165)
    plt.savefig(os.path.join(model_outpath, "feature_importance.png"), dpi=400)
    plt.savefig(os.path.join(model_outpath, "feature_importance.pdf"))


#===============================================================================
# TRAINING INFORMATION
#===============================================================================
df_training = pd.DataFrame(list(zip(iteration, train_acc, test_acc, train_loss, test_loss, adv_source_acc, adv_target_acc)),columns=["iteration", "train_acc", "test_acc", "train_loss", "test_loss", "adv_source_acc", "adv_target_acc"])
df_training.to_csv(os.path.join(model_outpath, 'training.csv'), index=False)

lr_info.to_csv(os.path.join(model_outpath, 'lr_info.csv'), index=False)

#iteration = df_training['iteration']
#train_acc = df_training['train_acc']
#test_acc = df_training['test_acc']
#train_loss = df_training['train_loss']
#test_loss = df_training['test_loss']
#adv_source_acc = df_training['adv_source_acc']
#adv_target_acc = df_training['adv_target_acc']
#adv_sum_acc = np.array(df_training['adv_source_acc']) + np.array(df_training['adv_target_acc'])

min_loss = np.amin(test_loss)
position = np.array(iteration[test_loss == min_loss])[0]



#-----------------------------------------------------------------------------------------------------------------
# Accuracy
#-----------------------------------------------------------------------------------------------------------------
fig1 = plt.figure(figsize=(9,5))
grid = [1, 1]
gs1 = gs.GridSpec(grid[0], grid[1])

ax1 = plt.subplot(gs1[0])
plt.axvline(position, color='grey')
plt.plot(iteration, train_acc, "-", color='red', label='Train (Class Accuracy)')
plt.plot(iteration, test_acc, "-", color='blue', label='Test (Class Accuracy)')
#plt.plot(iteration, adv_target_acc, "-", color='orange', label='Target (Domain Accuracy)')
#plt.plot(iteration, adv_source_acc, "-", color='green', label='Source (Domain Accuracy)')
#plt.plot(iteration, adv_sum_acc, "-", color='orchid', label='Sum (Domain Accuracy)')
plt.axhline(1, color='grey', linestyle='--')
ax1.set_xlabel("iterations", size=14, horizontalalignment='right', x=1.0)
ax1.set_ylabel("Accuracy", size=14, horizontalalignment='right', y=1.0)
ax1.tick_params(which='major', length=8)
ax1.tick_params(which='minor', length=4)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.grid(which='major', axis='x', linewidth=0.2, linestyle='-', color='0.75')
ax1.grid(which='major', axis='y', linewidth=0.2, linestyle='-', color='0.75')
ax1.spines['bottom'].set_linewidth(1)
ax1.spines['top'].set_linewidth(1)
ax1.spines['left'].set_linewidth(1)
ax1.spines['right'].set_linewidth(1)
ax1.margins(x=0)
ax1.set_ylim([0,1])
ax1.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False, loc='lower right')

ax3 = ax1.twiny()
ax3.set_xlabel("Learning rates", color='darkgreen', size=14, horizontalalignment='right', x=1.0)
ax3.tick_params('x', colors='darkgreen')
ax3.tick_params(which='major', length=8)
ax3.set_xlim([iteration[0],iteration[-1]])
ax3.set_xticks([iteration[0]]+lr_info["lr_last_it"].tolist()[:-1])
ax3.set_xticklabels(lr_info["lr_values"])

plt.subplots_adjust(left=0.09, bottom=0.115, right=0.97, top=0.9, wspace=0.18, hspace=0.165)
plt.savefig(os.path.join(model_outpath, "training_acc.png"), dpi=400)
plt.savefig(os.path.join(model_outpath, "training_acc.pdf"))

#-----------------------------------------------------------------------------------------------------------------
# Loss
#-----------------------------------------------------------------------------------------------------------------
fig1 = plt.figure(figsize=(9,5))
grid = [1, 1]
gs1 = gs.GridSpec(grid[0], grid[1])

ax2 = plt.subplot(gs1[0])
plt.axvline(position, color='grey')
plt.plot(iteration, train_loss, "-", color='red', label='Train (Class Loss)')
plt.plot(iteration, test_loss, "-", color='blue', label='Test (Class Loss)')
#plt.yscale('log')
ax2.set_xlabel("iterations", size=14, horizontalalignment='right', x=1.0)
ax2.set_ylabel("Loss", size=14, horizontalalignment='right', y=1.0)
ax2.tick_params(which='major', length=8)
ax2.tick_params(which='minor', length=4)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.grid(which='major', axis='x', linewidth=0.2, linestyle='-', color='0.75')
ax2.grid(which='major', axis='y', linewidth=0.2, linestyle='-', color='0.75')
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['top'].set_linewidth(1)
ax2.spines['left'].set_linewidth(1)
ax2.spines['right'].set_linewidth(1)
ax2.margins(x=0)
ax2.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False)

ax3 = ax2.twiny()
ax3.set_xlabel("Learning rates", color='darkgreen', size=14, horizontalalignment='right', x=1.0)
ax3.tick_params('x', colors='darkgreen')
ax3.tick_params(which='major', length=8)
ax3.set_xlim([iteration[0],iteration[-1]])
ax3.set_xticks([iteration[0]]+lr_info["lr_last_it"].tolist()[:-1])
ax3.set_xticklabels(lr_info["lr_values"])

plt.subplots_adjust(left=0.09, bottom=0.115, right=0.97, top=0.9, wspace=0.18, hspace=0.165)
plt.savefig(os.path.join(model_outpath, "training_loss.png"), dpi=400)
plt.savefig(os.path.join(model_outpath, "training_loss.pdf"))


#===============================================================================
# CHECK OVERTRAINING
#===============================================================================
print("")
print("------------------------------------------------------------------------")
print("Preparing overtraining plots")
print("------------------------------------------------------------------------")
for i_load in tqdm(range(num_load_for_check)):

    ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, class_colors, reweight_info = tools.get_sample(input_path, model[N][1], classes, N_signal, train_frac, load_size_training, i_load, reweight_info, features=variables+["evtWeight"], vec_features=vec_variables)

    train_data = tools.process_data(model_type, ds_full_train, vec_full_train, variables, vec_variables, var_use, vec_var_use)
    test_data = tools.process_data(model_type, ds_full_test, vec_full_test, variables, vec_variables, var_use, vec_var_use)

    ds_full_train = pd.DataFrame.from_dict(ds_full_train)
    ds_full_test = pd.DataFrame.from_dict(ds_full_test)

    n_eval_train_steps = int(len(train_data[-1])/eval_step_size) + 1
    n_eval_test_steps = int(len(test_data[-1])/eval_step_size) + 1

    class_model.eval()

    train_class_pred = []
    for i_eval in range(n_eval_train_steps):
        i_eval_output = tools.evaluate_model(model_type, train_data, class_model, i_eval, eval_step_size, None, None, stat_values, var_use, device, mode="predict")
        if i_eval_output is None:
            continue
        else:
            i_train_class_pred = i_eval_output
        train_class_pred = train_class_pred + i_train_class_pred.tolist()
    train_class_pred = np.array(train_class_pred)


    test_class_pred = []
    for i_eval in range(n_eval_test_steps):
        i_eval_output = tools.evaluate_model(model_type, test_data, class_model, i_eval, eval_step_size, None, None, stat_values, var_use, device, mode="predict")
        if i_eval_output is None:
            continue
        else:
            i_test_class_pred = i_eval_output
        test_class_pred = test_class_pred + i_test_class_pred.tolist()
    test_class_pred = np.array(test_class_pred)


    if model[N][4] == "cce":
        n_outputs = len(classes)
        for i in range(n_outputs):
            pred_name = 'score_C'+str(i)
            ds_full_test[pred_name] = test_class_pred[:,i]
            ds_full_train[pred_name] = train_class_pred[:,i]
    if model[N][4] == "bce":
        n_outputs = 1
        pred_name = 'score_C0'
        ds_full_test[pred_name] = 1 - test_class_pred[:,0]
        ds_full_train[pred_name] = 1 - train_class_pred[:,0]

    if i_load == 0:
        ds_check_test = ds_full_test.copy()
        ds_check_train = ds_full_train.copy()
    else:
        ds_check_test = pd.concat([ds_check_test, ds_full_test])
        ds_check_train = pd.concat([ds_check_train, ds_full_train])


for i in range(n_outputs):
    fig1 = plt.figure(figsize=(9,5))
    gs1 = gs.GridSpec(1,1)

    #==================================================
    ax1 = plt.subplot(gs1[0])
    #==================================================
    var = 'score_C'+str(i)
    bins = np.linspace(0,1,51)
    yTrain = []
    errTrain = []
    yTest = []
    errTest = []
    for ikey in range(len(class_names)):
        yH, errH = tools.step_plot( ax1, var, ds_full_train[ds_full_train["class"] == ikey], label=class_labels[ikey]+" (train)", color=class_colors[ikey], weight="mvaWeight", bins=bins, error=True )
        yTrain.append(yH)
        errTrain.append(errH)
        yH, errH = tools.step_plot( ax1, var, ds_full_test[ds_full_test["class"] == ikey], label=class_labels[ikey]+" (test)", color=class_colors[ikey], weight="mvaWeight", bins=bins, error=True, linestyle='dotted' )
        yTest.append(yH)
        errTest.append(errH)
    ax1.set_ylabel("Events normalized", size=14, horizontalalignment='right', y=1.0)
    ax1.set_xlabel(class_names[i] + " score", size=14, horizontalalignment='right', x=1.0)
    plt.yscale('log')
    ax1.set_ylim([1.E-6,1.])

    ax1.tick_params(which='major', length=8)
    ax1.tick_params(which='minor', length=4)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    #ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)
    ax1.margins(x=0)
    ax1.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False, loc='upper center')

    plt.subplots_adjust(left=0.09, bottom=0.115, right=0.97, top=0.95, wspace=0.18, hspace=0.165)
    plt.savefig(os.path.join(model_outpath, var + "_hist.png"), dpi=400)
    plt.savefig(os.path.join(model_outpath, var + "_hist.pdf"))


    #------------------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(9,5))
    grid = [1, 1]
    gs1 = gs.GridSpec(grid[0], grid[1])
    #==================================================
    ax1 = plt.subplot(gs1[0])
    #==================================================
    var = 'score_C'+str(i)
    signal_train_roc = []
    signal_test_roc = []
    bkg_train_roc = []
    bkg_test_roc = []
    for ikey in range(len(class_names)):
        if ikey == i:
            signal_train_roc.append(ds_full_train[ds_full_train["class"] == ikey])
            signal_test_roc.append(ds_full_test[ds_full_test["class"] == ikey])
        else:
            bkg_train_roc.append(ds_full_train[ds_full_train["class"] == ikey])
            bkg_test_roc.append(ds_full_test[ds_full_test["class"] == ikey])

    ctr_train = tools.control( var, signal_train_roc, bkg_train_roc, weight="evtWeight", bins=np.linspace(0,1,1001) )
    ctr_train.roc_plot(label='ROC (train)', color='blue', linestyle="-", version=2)
    ctr_test = tools.control( var, signal_test_roc, bkg_test_roc, weight="evtWeight", bins=np.linspace(0,1,1001) )
    ctr_test.roc_plot(label='ROC (test)', color='blue', linestyle="--", version=2)

    ax1.set_xlabel("Signal efficiency", size=14, horizontalalignment='right', x=1.0)
    ax1.set_ylabel("Background efficiency", size=14, horizontalalignment='right', y=1.0)
    plt.yscale('log')
    ax1.set_xlim([0,1])
    ax1.set_ylim([1.E-3,1.])

    ax1.grid(which='major', axis='x', linewidth=0.2, linestyle='-', color='0.75')
    ax1.grid(which='major', axis='y', linewidth=0.2, linestyle='-', color='0.75')
    ax1.tick_params(which='major', length=8)
    ax1.tick_params(which='minor', length=4)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    #ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)
    ax1.margins(x=0)
    ax1.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False, loc='lower center')

    plt.subplots_adjust(left=0.09, bottom=0.115, right=0.97, top=0.95, wspace=0.18, hspace=0.165)
    plt.savefig(os.path.join(model_outpath, var + "_roc.png"), dpi=400)
    plt.savefig(os.path.join(model_outpath, var + "_roc.pdf"))

del ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, class_colors


#===============================================================================
# SAVE MODEL
#===============================================================================
tools.save_model(model_type, class_model, model_outpath, stat_values["dim"], device)



#===============================================================================
end = time.time()
hours = int((end - start)/3600)
minutes = int(((end - start)%3600)/60)
seconds = int(((end - start)%3600)%60)

print("")
print("-----------------------------------------------------------------------------------")
print("Total process duration: " + str(hours) + " hours " + str(minutes) + " minutes " + str(seconds) + " seconds")
print("-----------------------------------------------------------------------------------")
print("")

