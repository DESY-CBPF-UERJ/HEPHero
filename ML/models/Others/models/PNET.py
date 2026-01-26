import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.ticker import AutoMinorLocator
import torch
import torch.nn as nn
import sys
import os
from tqdm import tqdm
from statsmodels.stats.weightstats import DescrStatsW
from .ParticleNet import ParticleNetTagger
sys.path.append("..")
from custom_opts.ranger import Ranger
import tools

#==================================================================================================
def build_PNET(vec_variables, vec_var_use, n_classes, parameters, stat_values, device):

    conv_params, fc_params = parameters[7]

    """
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
        ]
    fc_params = [(256, 0.1)]

    conv_params = [
        (8, (64, 64, 64)),
        (8, (96, 96, 96)),
        (8, (128, 128, 128)),
    ]
    fc_params = [(128, 0.1)]
    """

    vec_var_info = {}
    for i in range(len(vec_variables)):
        var_key = vec_variables[i].split("_")[0]
        if "F" in vec_var_use[i]:
            if var_key in vec_var_info:
                vec_var_info[var_key] = vec_var_info[var_key] + 1
            else:
                vec_var_info[var_key] = 1

    pf_features_dims = vec_var_info['jetPFcand']
    sv_features_dims = vec_var_info['jetSV']
    num_classes = n_classes

    model = ParticleNetTagger(pf_features_dims, sv_features_dims, num_classes,
                              parameters, stat_values, device, conv_params, fc_params,
                              use_fusion=True,
                              use_fts_bn=False,
                              use_counts=True,
                              pf_input_dropout=None,
                              sv_input_dropout=None,
                              for_inference=False
                              )

    return model


#==================================================================================================
def model_parameters_PNET(param_dict):
    conv_p = param_dict["conv_params"]
    fc_p = param_dict["fc_params"]

    conv_params = [
        (conv_p[0], (conv_p[1], conv_p[1], conv_p[1])),
        (conv_p[0], (conv_p[2], conv_p[2], conv_p[2])),
        (conv_p[0], (conv_p[3], conv_p[3], conv_p[3])),
        ]

    fc_params = [(fc_p[0], fc_p[1])]

    return [[conv_params, fc_params]]


#==================================================================================================
def features_stat_PNET(train_data, test_data, vec_train_data, vec_test_data, vec_variables, vec_var_names, vec_var_use, class_names, class_labels, class_colors, plots_outpath, load_it=None):

    vec_train_info = {}
    vec_test_info = {}
    for var in vec_variables:
        vec_train_info[var.split("_")[0]] = vec_train_data[var].shape
        vec_test_info[var.split("_")[0]] = vec_test_data[var].shape
        vec_train_data[var] = vec_train_data[var].flatten()
        vec_test_data[var] = vec_test_data[var].flatten()

    vec_types = list(vec_train_info.keys())
    vec_train_shapes = list(vec_train_info.values())
    vec_test_shapes = list(vec_test_info.values())

    ds_mask_train = []
    ds_mask_test = []
    for k in range(len(vec_types)):
        variables_type_k = [var for var in vec_variables if var.split("_")[0] == vec_types[k]]

        ds_mask_train.append(np.abs(np.stack([vec_train_data[key] for key in vec_train_data.keys() if key in variables_type_k], axis=1)).sum(axis=1, keepdims=False) != 0)
        ds_mask_test.append(np.abs(np.stack([vec_test_data[key] for key in vec_test_data.keys() if key in variables_type_k], axis=1)).sum(axis=1, keepdims=False) != 0)


    vec_mean = []
    vec_std = []
    dim = []
    for k in range(len(vec_types)):
        type_mean = []
        type_std = []
        for i in range(len(vec_variables)):
            if vec_variables[i].split("_")[0] == vec_types[k]:
                data_var = {vec_variables[i]: vec_train_data[vec_variables[i]], 'mvaWeight': ((np.ones(vec_train_shapes[k]).T*np.array(train_data["mvaWeight"])).T).flatten(), 'mask': ds_mask_train[k]}
                data_var = pd.DataFrame.from_dict(data_var)
                weighted_stats = DescrStatsW(data_var[data_var["mask"] != 0][vec_variables[i]], weights=data_var[data_var["mask"] != 0]["mvaWeight"], ddof=0)
                type_mean.append(weighted_stats.mean)
                type_std.append(weighted_stats.std)
                del data_var
        vec_mean.append(type_mean)
        vec_std.append(type_std)
        dim.append((len(vec_mean[k]),vec_train_shapes[k][1]))
    np.set_printoptions(legacy='1.25')
    if load_it is None:
        print("mean: " + str(vec_mean))
        print("std: " + str(vec_std))
        print("dim: " + str(dim))
    stat_values={"mean": vec_mean, "std": vec_std, "dim": dim}


    for k in range(len(vec_types)):
        kvar = 0
        for ivar in range(len(vec_variables)):
            if vec_variables[ivar].split("_")[0] == vec_types[k]:

                fig1 = plt.figure(figsize=(9,5))
                gs1 = gs.GridSpec(1, 1)
                #==================================================
                ax1 = plt.subplot(gs1[0])
                #==================================================
                var = vec_variables[ivar]
                bins = np.linspace(vec_mean[k][kvar]-5*vec_std[k][kvar],vec_mean[k][kvar]+5*vec_std[k][kvar],51)
                for ikey in range(len(class_names)):

                    data_var = {vec_variables[ivar]: vec_train_data[vec_variables[ivar]], 'mvaWeight': ((np.ones(vec_train_shapes[k]).T*np.array(train_data["mvaWeight"])).T).flatten(), 'class': ((np.ones(vec_train_shapes[k]).T*np.array(train_data["class"])).T).flatten(), 'mask': ds_mask_train[k]}
                    data_var = pd.DataFrame.from_dict(data_var)
                    tools.step_plot( ax1, var, data_var[(data_var["class"] == ikey) & (data_var["mask"] != 0)], label=class_labels[ikey]+" (train)", color=class_colors[ikey], weight="mvaWeight", bins=bins, error=True )

                    data_var = {vec_variables[ivar]: vec_test_data[vec_variables[ivar]], 'mvaWeight': ((np.ones(vec_test_shapes[k]).T*np.array(test_data["mvaWeight"])).T).flatten(), 'class': ((np.ones(vec_test_shapes[k]).T*np.array(test_data["class"])).T).flatten(), 'mask': ds_mask_test[k]}
                    data_var = pd.DataFrame.from_dict(data_var)
                    tools.step_plot( ax1, var, data_var[(data_var["class"] == ikey) & (data_var["mask"] != 0)], label=class_labels[ikey]+" (test)", color=class_colors[ikey], weight="mvaWeight", bins=bins, error=True, linestyle='dotted' )

                    del data_var

                ax1.set_xlabel(vec_var_names[ivar], size=14, horizontalalignment='right', x=1.0)
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
                if load_it is None:
                    plt.savefig(os.path.join(plots_outpath, var + '.png'), dpi=400)
                    plt.savefig(os.path.join(plots_outpath, var + '.pdf'))
                else:
                    plt.savefig(os.path.join(plots_outpath, var +"_"+ str(load_it) + '.png'), dpi=400)
                    plt.savefig(os.path.join(plots_outpath, var +"_"+ str(load_it) + '.pdf'))
                plt.close()

                kvar += 1


    return stat_values


#==================================================================================================
def update_PNET(model, criterion, parameters, batch_data, device):

    if parameters[3] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters[6], eps=1e-07)
        # lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
    elif parameters[3] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=parameters[6])
        # lr=?, momentum=0, dampening=0, weight_decay=0, nesterov=False
    elif parameters[3] == "ranger":
        optimizer = Ranger(model.parameters(), lr=parameters[6])
    #print(optimizer.state_dict())

    pf_points, pf_features, sv_points, sv_features, data_y_b, data_w_b = batch_data
    del batch_data

    if device == "cuda":
        w = torch.FloatTensor(data_w_b).view(-1,1).to("cuda")
        y = torch.IntTensor(data_y_b).view(-1,1).to("cuda")
        pf_points = pf_points.to("cuda")
        pf_features = pf_features.to("cuda")
        sv_points = sv_points.to("cuda")
        sv_features = sv_features.to("cuda")
    else:
        w = torch.FloatTensor(data_w_b).view(-1,1)
        y = torch.IntTensor(data_y_b).view(-1,1)

    pf_points.requires_grad=True
    pf_features.requires_grad=True
    sv_points.requires_grad=True
    sv_features.requires_grad=True
    pf_mask = (pf_features.abs().sum(dim=1, keepdim=True) != 0)
    sv_mask = (sv_features.abs().sum(dim=1, keepdim=True) != 0)

    yhat = model(pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask)

    loss = criterion(y, yhat, w, device=device)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return model


#==================================================================================================
def process_data_PNET(scalar_var, vector_var, vec_variables, vec_var_use):

    data_y = np.array(scalar_var['class']).ravel()
    data_w = np.array(scalar_var['mvaWeight']).ravel()

    data_pf_points_list = []
    data_pf_features_list = []
    data_sv_points_list = []
    data_sv_features_list = []
    for ivar in range(len(vec_variables)):
        if vec_variables[ivar].split("_")[0] == 'jetPFcand':
            if "P" in vec_var_use[ivar]:
                data_pf_points_list.append(vector_var[vec_variables[ivar]])
            if "F" in vec_var_use[ivar]:
                data_pf_features_list.append(vector_var[vec_variables[ivar]])
        elif vec_variables[ivar].split("_")[0] == 'jetSV':
            if "P" in vec_var_use[ivar]:
                data_sv_points_list.append(vector_var[vec_variables[ivar]])
            if "F" in vec_var_use[ivar]:
                data_sv_features_list.append(vector_var[vec_variables[ivar]])
        del vector_var[vec_variables[ivar]]

    data_pf_points = torch.FloatTensor(np.stack(data_pf_points_list, axis=1))
    del data_pf_points_list
    data_pf_features = torch.FloatTensor(np.stack(data_pf_features_list, axis=1))
    del data_pf_features_list
    data_sv_points = torch.FloatTensor(np.stack(data_sv_points_list, axis=1))
    del data_sv_points_list
    data_sv_features = torch.FloatTensor(np.stack(data_sv_features_list, axis=1))
    del data_sv_features_list

    input_data = [data_pf_points, data_pf_features, data_sv_points, data_sv_features, data_y, data_w]

    return input_data


#==================================================================================================
def evaluate_PNET(input_data, model, i_eval, eval_step_size, criterion, parameters, device, mode):

    data_pf_points, data_pf_features, data_sv_points, data_sv_features, data_y, data_w = input_data

    n_eval_steps = int(len(data_w)/eval_step_size) + 1
    last_eval_step = len(data_w)%eval_step_size

    if i_eval < n_eval_steps-1:
        eval_data_pf_points = data_pf_points[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
        eval_data_pf_features = data_pf_features[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
        eval_data_sv_points = data_sv_points[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
        eval_data_sv_features = data_sv_features[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
        eval_data_y = data_y[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
        eval_data_w = data_w[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
    elif last_eval_step > 0:
        eval_data_pf_points = data_pf_points[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_step]
        eval_data_pf_features = data_pf_features[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_step]
        eval_data_sv_points = data_sv_points[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_step]
        eval_data_sv_features = data_sv_features[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_step]
        eval_data_y = data_y[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_step]
        eval_data_w = data_w[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_step]
    else:
        return None

    eval_data_pf_mask = (eval_data_pf_features.abs().sum(dim=1, keepdim=True) != 0)
    eval_data_sv_mask = (eval_data_sv_features.abs().sum(dim=1, keepdim=True) != 0)

    if device == "cuda":
        eval_data_yhat = model(eval_data_pf_points.to("cuda"), eval_data_pf_features.to("cuda"), eval_data_pf_mask.to("cuda"), eval_data_sv_points.to("cuda"), eval_data_sv_features.to("cuda"), eval_data_sv_mask.to("cuda"))
        eval_data_yhat = eval_data_yhat.cpu()
    else:
        eval_data_yhat = model(eval_data_pf_points, eval_data_pf_features, eval_data_pf_mask, eval_data_sv_points, eval_data_sv_features, eval_data_sv_mask)

    if mode == "predict":
        return eval_data_yhat

    elif mode == "metric":
        eval_data_w_sum = eval_data_w.sum()
        data_loss_i = eval_data_w_sum*criterion(torch.IntTensor(eval_data_y).view(-1,1), eval_data_yhat, torch.FloatTensor(eval_data_w).view(-1,1)).item()
        if parameters[4] == 'cce':
            data_acc_i = eval_data_w_sum*np.average(eval_data_y == eval_data_yhat.max(1)[1].numpy(), weights=eval_data_w)
        elif parameters[4] == 'bce':
            data_acc_i = eval_data_w_sum*np.average(eval_data_y == (eval_data_yhat[:, 0] > 0.5).numpy(), weights=eval_data_w)
        del eval_data_yhat

        return [data_loss_i, data_acc_i]


#==================================================================================================
def feature_score_PNET(input_data, model, min_loss, eval_step_size, criterion, parameters, vec_variables, vec_var_use, vec_var_names, device):

    data_pf_points, data_pf_features, data_sv_points, data_sv_features, data_y, data_w = input_data
    n_entries = len(input_data[-1])
    n_eval_steps = int(len(input_data[-1])/eval_step_size) + 1
    data_w_sum = input_data[-1].sum()

    ipf = 0
    isv = 0
    features_score = []
    features_score_unc = []
    features_names = []
    for ivar in tqdm(range(len(vec_variables))): # use vec_types in the future
        losses = []
        for irep in range(10):

            data_pf_features_shuffled = data_pf_features.detach().clone()
            data_sv_features_shuffled = data_sv_features.detach().clone()
            p_idx = np.random.permutation(n_entries)
            if vec_variables[ivar].split("_")[0] == 'jetPFcand':
                if "F" in vec_var_use[ivar]:
                    data_pf_features_shuffled[:,ipf] = data_pf_features_shuffled[:,ipf][p_idx]
                    if irep == 9:
                        ipf += 1
            elif vec_variables[ivar].split("_")[0] == 'jetSV':
                if "F" in vec_var_use[ivar]:
                    data_sv_features_shuffled[:,isv] = data_sv_features_shuffled[:,isv][p_idx]
                    if irep == 9:
                        isv += 1
            shuffled_data = data_pf_points, data_pf_features_shuffled, data_sv_points, data_sv_features_shuffled, data_y, data_w

            data_loss_i = 0
            for i_eval in range(n_eval_steps):
                i_eval_output = evaluate_PNET(shuffled_data, model, i_eval, eval_step_size, criterion, parameters, device, mode="metric")
                if i_eval_output is None:
                    continue
                else:
                    i_eval_loss, i_eval_acc = i_eval_output
                data_loss_i += i_eval_loss
            data_loss_i = data_loss_i/data_w_sum

            losses.append(data_loss_i)

        losses = np.array(losses)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        features_score.append(np.around((mean_loss - min_loss)/np.abs(min_loss), decimals=3))
        features_score_unc.append(np.around(std_loss/np.abs(min_loss), decimals=3))
        features_names.append(vec_var_names[ivar])

    feature_score_info = features_score, features_score_unc, features_names

    return feature_score_info


#==================================================================================================
def save_PNET(model, model_outpath, dim, device):

    #ONNX
    input_names = ['pf_points', 'pf_features', 'pf_mask', 'sv_points', 'sv_features', 'sv_mask']
    output_names = ['softmax']
    dynamic_axes = {'pf_points': {0: 'N', 2: 'n_pf'}, 'pf_features': {0: 'N', 2: 'n_pf'}, 'pf_mask': {0: 'N', 2: 'n_pf'}, 'sv_points': {0: 'N', 2: 'n_sv'}, 'sv_features': {0: 'N', 2: 'n_sv'}, 'sv_mask': {0: 'N', 2: 'n_sv'}, 'softmax': {0: 'N'}}
    input_shapes = {'pf_points': (1, 2, dim[0][1]), 'pf_features': (1, dim[0][0], dim[0][1]), 'pf_mask': (1, 1, dim[0][1]), 'sv_points': (1, 2, dim[1][1]), 'sv_features': (1, dim[1][0], dim[1][1]), 'sv_mask': (1, 1, dim[1][1])}
    inputs = tuple(torch.ones(input_shapes[k], dtype=torch.float32) for k in input_names)

    if device == "cuda":
        model = model.cpu()
        model.module.stat_device('cpu')

        torch.save(model.module.state_dict(), os.path.join(model_outpath, "model_state_dict.pt"))

        torch.onnx.export(model.module, inputs, os.path.join(model_outpath, "model.onnx"),
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        opset_version=15)
    else:
        torch.save(model.state_dict(), os.path.join(model_outpath, "model_state_dict.pt"))

        torch.onnx.export(model, inputs, os.path.join(model_outpath, "model.onnx"),
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        opset_version=15)






"""
pf_points = torch.FloatTensor([[[-1.6944e-01, -1.6908e-01, -2.4672e-01, -2.4452e-01, -2.7565e-01,
        -1.5864e-01, -2.6961e-01, -2.0076e-01, -1.2293e-01, -3.2472e-01,
        -3.1520e-01, -3.9595e-01, -1.4289e-01, -4.3862e-01, -2.7455e-01,
        -2.5532e-01, -3.0550e-01, -4.1078e-01, -2.7473e-01, -2.9213e-01,
        -3.3992e-01, -1.3484e-01, -3.2417e-01, -7.9697e-01,  1.4258e-01,
        -2.2731e-01,  1.6413e-02,  1.2121e+00,  2.2424e-01, -3.9943e-01,
        -4.0900e-02,  8.8143e-01,  2.8467e-01, -9.7037e-01, -3.7398e-01,
        -3.7892e-01,  1.0178e+00, -5.4903e-01,  7.3641e-01,  5.8314e-01,
        -1.1927e-01,  4.1505e-01,  7.8548e-01, -2.4122e-01,  1.8245e-02,
        1.6711e-01, -4.5949e-01,  2.3486e-01, -3.6867e-01,  7.3201e-01],
        [-3.0759e-02, -1.2497e-02, -3.9308e-01, -3.9190e-01, -4.0362e-01,
        -1.8943e-02, -4.0382e-01, -4.0059e-01,  6.8622e-04, -3.5714e-01,
        1.4171e-01, -3.1407e-01, -3.4991e-01, -4.2286e-01, -3.7325e-01,
        -4.4308e-01, -3.8517e-01, -3.5655e-01, -4.0304e-01, -4.8263e-01,
        -2.9844e-01, -6.6402e-02,  9.7470e-02,  5.3860e-01,  1.1666e+00,
        -1.3665e-02, -4.6283e-02,  9.7221e-01,  3.3917e-01, -4.3410e-01,
        2.7316e-01,  1.2253e+00,  1.0661e+00,  5.2424e-01,  8.3460e-01,
        5.4251e-01,  2.3692e-01,  9.1643e-01,  5.8898e-01,  1.3585e+00,
        -3.0938e-01,  3.5567e-01,  2.1162e-01, -5.1885e-01,  4.1330e-01,
        8.6449e-01, -4.7619e-01,  1.2515e+00, -2.1000e-02,  4.9630e-01]],

        [[-4.9961e-02, -5.1975e-02, -4.4834e-02, -5.3990e-02, -5.8750e-02,
        -4.7214e-02, -5.3074e-02, -6.4976e-02, -8.0358e-02,  3.1157e-02,
        -7.7794e-02,  8.2794e-02,  1.7973e-02, -3.4946e-02, -1.0929e-01,
        -3.8608e-02,  1.3700e-01, -1.4152e-01,  8.9386e-02, -9.0429e-02,
        1.2663e-02,  9.6162e-02,  8.6457e-02,  6.5582e-02,  2.3130e-01,
        2.4943e-01,  2.3514e-01,  3.9701e-01,  4.3657e-01,  1.2509e-01,
        2.4869e-01,  7.2496e-01,  1.4285e-01, -3.3664e-02,  3.1461e-01,
        -1.1863e-01, -5.9666e-02, -5.6370e-02,  4.1020e-01,  4.5440e-02,
        6.8871e-01, -1.6093e-01,  5.0487e-01,  4.0873e-01, -1.0618e-01,
        -1.8473e-01,  5.3380e-01,  5.2959e-01,  5.0395e-01,  2.1244e-01],
        [-3.2101e-02, -3.6300e-02, -3.3858e-02, -2.4971e-02, -1.6768e-02,
        -5.8664e-02, -4.1183e-02,  1.0284e-02,  2.6104e-02, -7.0187e-02,
        -6.9112e-02, -7.2335e-02, -1.2507e-01, -4.2161e-02,  1.2626e-02,
        -6.6088e-02,  3.8441e-01, -1.4763e-01, -5.5340e-04, -1.7195e-01,
        -8.7183e-02, -1.7498e-01, -1.6696e-01,  3.0707e-01,  8.4165e-01,
        5.2613e-01,  6.4380e-01,  7.6295e-02,  1.2503e-01,  5.3716e-01,
        4.0912e-01,  2.4292e-01,  7.4996e-01, -1.7956e-01,  6.5864e-01,
        -1.0631e-01,  1.0585e-02,  7.0074e-01,  2.8872e-01, -1.8268e-01,
        7.4357e-02,  6.2748e-01,  1.1194e-01,  5.0131e-01, -2.0614e-01,
        -2.0135e-01,  5.2994e-01,  7.1803e-01,  8.1676e-01,  2.6333e-01]],

        [[-7.5982e-02, -6.8108e-02, -8.8068e-02, -1.9950e-02, -5.0713e-02,
        -2.8740e-02, -2.5627e-02,  2.4912e-02, -1.2139e-01, -1.1279e-01,
        -1.0345e-01, -3.2219e-02, -2.2156e-01, -2.2148e-02,  9.1747e-02,
        -1.1846e-01, -5.8404e-02,  6.7862e-01,  2.2524e-01,  6.8301e-01,
        5.6381e-01,  5.5831e-01,  6.4200e-01, -6.1700e-02, -3.5376e-01,
        1.3844e-01, -3.1366e-01,  2.0151e-02,  8.4057e-02,  1.8349e-01,
        -1.6534e-01, -5.8037e-02, -3.3454e-01, -1.0473e-01,  6.9839e-01,
        2.9940e-01, -1.7936e-02,  7.8380e-02,  3.1331e-01,  2.5325e-01,
        -2.7576e-01, -1.0949e-01,  6.4786e-01, -2.0379e-01,  3.4938e-01,
        6.6008e-03,  3.2851e-01, -1.9995e-01,  1.7158e-01, -2.6359e-02],
        [ 7.1875e-02,  7.0312e-02,  7.8208e-02,  2.5790e-01,  2.4638e-01,
        4.5311e-02,  2.9931e-01,  2.2763e-01,  4.3748e-02,  9.6176e-02,
        2.2303e-01,  1.2597e-01, -6.2995e-02, -6.9387e-01,  2.3798e-01,
        -2.2754e-02,  2.8037e-01, -5.1877e-01, -8.5452e-01, -7.5735e-01,
        -7.8352e-01, -8.0071e-01, -6.1048e-01, -7.2258e-01, -8.0070e-01,
        -1.5040e-01,  2.8367e-03,  3.0029e-01, -1.1360e+00, -1.8860e-02,
        1.8367e-01, -2.5459e-01, -3.1201e-01, -4.0567e-01, -3.3457e-01,
        -1.0039e-01, -1.2557e+00, -4.6965e-01, -6.7327e-01, -1.2008e+00,
        -4.0596e-01, -6.5646e-01, -4.2087e-02, -5.4962e-01, -4.9620e-01,
        -9.6233e-01, -1.9670e-01, -6.6125e-02, -6.1272e-01, -7.3099e-01]],

        [[ 1.6225e-01,  1.6921e-01,  2.6277e-01,  1.6444e-01,  1.6280e-01,
        1.7378e-01,  1.5364e-01,  2.0802e-01,  1.6921e-01,  9.9440e-02,
        -8.7599e-01, -8.4779e-01,  1.2911e+00,  1.9154e-01,  1.4705e-01,
        -7.4910e-01,  4.3728e-01, -1.2418e+00,  2.1517e-01,  8.7904e-02,
        1.3805e+00,  1.6509e+00, -9.6187e-01,  3.6532e-01,  7.1607e-02,
        8.0661e-01,  4.1988e-01,  1.9603e-02,  1.6884e-01,  8.3143e-02,
        1.0566e+00,  1.1196e+00,  1.5070e+00,  1.7050e+00, -9.0456e-01,
        1.3794e+00, -2.4710e-02,  1.1304e+00,  8.7528e-01, -1.1950e+00,
        1.4682e+00,  1.6830e+00,  1.0848e+00,  1.8013e+00, -8.9723e-01,
        -7.7931e-01,  1.5575e-02,  4.8818e-01,  3.6202e-01, -5.8228e-01],
        [-2.1986e-01, -2.2523e-01, -9.0360e-02, -2.4173e-01, -1.7083e-01,
        -2.3001e-01, -2.8754e-01, -2.0218e-01, -1.9837e-01, -1.2396e-01,
        1.2576e+00,  2.2293e-01, -1.1937e-02, -2.3167e-01, -1.7289e-01,
        5.2411e-01,  4.5995e-01,  4.3690e-01,  3.2928e-01, -2.2864e-01,
        -3.6967e-01,  6.0528e-02,  1.3142e+00,  1.3249e-01, -2.7298e-01,
        2.3661e-01,  1.0076e-01, -3.7415e-01, -3.7631e-02,  6.4911e-01,
        -8.9190e-03,  6.1904e-01, -5.4447e-01, -1.0609e-01,  1.4284e-01,
        -4.7894e-01, -1.8088e-01,  6.6016e-01,  9.9853e-01,  6.6092e-01,
        -3.9555e-01,  2.2810e-01,  7.5284e-01,  3.7665e-01,  1.0564e+00,
        -1.7981e-01,  6.3701e-01,  7.0741e-01,  8.1024e-01,  7.6317e-01]],

        [[-5.3100e-01, -6.6650e-01, -5.0847e-01, -6.0076e-01, -5.6725e-01,
        -7.0898e-01, -7.2546e-01, -6.2164e-01, -5.2770e-01, -8.9319e-01,
        -7.4652e-01, -5.7274e-01, -9.1095e-01, -4.2845e-01, -2.2978e-01,
        -7.4707e-01, -7.5879e-01, -9.1461e-01,  3.9945e-02, -8.4320e-01,
        -7.0623e-01, -6.3702e-01, -3.4148e-01, -5.5938e-01, -6.4709e-01,
        1.2166e+00,  7.1819e-01,  2.8275e-01,  7.8612e-01, -2.5138e-01,
        7.5829e-01,  4.9699e-01,  9.5696e-01,  7.1086e-01, -6.7309e-01,
        -9.9848e-01, -5.3869e-01,  8.7365e-01,  1.3856e+00, -6.4672e-01,
        -5.9527e-01, -1.2412e-01, -6.1193e-01, -6.0680e-01,  2.4173e-01,
        -9.3933e-01,  5.9807e-01,  7.0464e-01, -2.2062e-01,  9.3371e-01],
        [ 1.8448e-01,  1.6963e-01,  2.1133e-01,  2.0860e-01,  1.7608e-01,
        2.1485e-01,  2.6114e-01,  2.3994e-01,  1.4502e-01,  1.9492e-01,
        1.3027e-01,  1.7452e-01,  2.0996e-01,  2.6035e-01, -6.2942e-01,
        2.2979e-01,  1.6709e-01,  3.2960e-01, -4.6848e-01,  3.2657e-01,
        2.1787e-01,  1.5097e-01, -1.5907e+00, -1.4102e+00,  7.9291e-02,
        -1.0000e-01, -3.7638e-01, -1.2456e+00,  6.7433e-01, -9.8870e-01,
        -2.8760e-01, -5.7160e-01,  4.2189e-01,  1.8574e-01,  4.0225e-01,
        -1.6618e+00,  2.7881e-01,  5.4825e-01,  2.7687e-01, -1.0495e+00,
        4.4074e-01, -7.8244e-01,  2.8957e-01, -1.3689e+00, -9.1155e-01,
        4.5595e-02, -6.0599e-01,  1.9853e-01, -1.3889e+00, -3.1358e-01]]])

pf_features = torch.FloatTensor([[[ 4.1372e+00,  3.8163e+00,  3.6747e+00,  3.1016e+00,  3.0939e+00,
        3.0460e+00,  3.0348e+00,  2.6507e+00,  2.4764e+00,  2.3850e+00,
        2.3439e+00,  2.1998e+00,  2.1215e+00,  1.9997e+00,  1.8536e+00,
        1.7258e+00,  1.6102e+00,  1.4524e+00,  1.4085e+00,  1.2561e+00,
        1.2187e+00,  1.1983e+00,  1.1953e+00,  1.0848e+00,  9.9296e-01,
        9.4403e-01,  9.3640e-01,  9.1941e-01,  8.9177e-01,  8.0134e-01,
        7.4454e-01,  7.2202e-01,  6.6392e-01,  6.5279e-01,  6.4821e-01,
        6.4309e-01,  6.3069e-01,  6.1814e-01,  6.0329e-01,  5.7182e-01,
        5.5233e-01,  5.2556e-01,  4.7888e-01,  4.6241e-01,  4.5624e-01,
        4.5500e-01,  4.0024e-01,  3.8906e-01,  3.7237e-01,  3.5129e-01],
        [ 4.2036e+00,  3.8829e+00,  3.7165e+00,  3.1441e+00,  3.1279e+00,
        3.1163e+00,  3.0704e+00,  2.7065e+00,  2.5601e+00,  2.4077e+00,
        2.3685e+00,  2.2100e+00,  2.1977e+00,  2.0048e+00,  1.8881e+00,
        1.7655e+00,  1.6373e+00,  1.4610e+00,  1.4433e+00,  1.2867e+00,
        1.2390e+00,  1.2774e+00,  1.2189e+00,  1.1180e+00,  1.2094e+00,
        9.9149e-01,  1.0839e+00,  2.0060e+00,  1.1578e+00,  8.1281e-01,
        8.6325e-01,  1.5055e+00,  9.7117e-01,  7.4574e-01,  6.6416e-01,
        6.5828e-01,  1.5373e+00,  6.1820e-01,  1.2605e+00,  1.1017e+00,
        6.3750e-01,  9.2549e-01,  1.1785e+00,  5.0930e-01,  6.0640e-01,
        6.8775e-01,  4.0764e-01,  6.6449e-01,  3.9109e-01,  1.0039e+00],
        [-1.6944e-01, -1.6908e-01, -2.4672e-01, -2.4452e-01, -2.7565e-01,
        -1.5864e-01, -2.6961e-01, -2.0076e-01, -1.2293e-01, -3.2472e-01,
        -3.1520e-01, -3.9595e-01, -1.4289e-01, -4.3862e-01, -2.7455e-01,
        -2.5532e-01, -3.0550e-01, -4.1078e-01, -2.7473e-01, -2.9213e-01,
        -3.3992e-01, -1.3484e-01, -3.2417e-01, -7.9697e-01,  1.4258e-01,
        -2.2731e-01,  1.6413e-02,  1.2121e+00,  2.2424e-01, -3.9943e-01,
        -4.0900e-02,  8.8143e-01,  2.8467e-01, -9.7037e-01, -3.7398e-01,
        -3.7892e-01,  1.0178e+00, -5.4903e-01,  7.3641e-01,  5.8314e-01,
        -1.1927e-01,  4.1505e-01,  7.8548e-01, -2.4122e-01,  1.8245e-02,
        1.6711e-01, -4.5949e-01,  2.3486e-01, -3.6867e-01,  7.3201e-01],
        [-3.0759e-02, -1.2497e-02, -3.9308e-01, -3.9190e-01, -4.0362e-01,
        -1.8943e-02, -4.0382e-01, -4.0059e-01,  6.8622e-04, -3.5714e-01,
        1.4171e-01, -3.1407e-01, -3.4991e-01, -4.2286e-01, -3.7325e-01,
        -4.4308e-01, -3.8517e-01, -3.5655e-01, -4.0304e-01, -4.8263e-01,
        -2.9844e-01, -6.6402e-02,  9.7470e-02,  5.3860e-01,  1.1666e+00,
        -1.3665e-02, -4.6283e-02,  9.7221e-01,  3.3917e-01, -4.3410e-01,
        2.7316e-01,  1.2253e+00,  1.0661e+00,  5.2424e-01,  8.3460e-01,
        5.4251e-01,  2.3692e-01,  9.1643e-01,  5.8898e-01,  1.3585e+00,
        -3.0938e-01,  3.5567e-01,  2.1162e-01, -5.1885e-01,  4.1330e-01,
        8.6449e-01, -4.7619e-01,  1.2515e+00, -2.1000e-02,  4.9630e-01]],

        [[ 4.8157e+00,  4.7340e+00,  4.5565e+00,  4.0337e+00,  3.6936e+00,
        2.7996e+00,  2.6955e+00,  2.6205e+00,  2.5810e+00,  2.5119e+00,
        2.3150e+00,  2.2587e+00,  1.9336e+00,  1.7950e+00,  1.6605e+00,
        1.5785e+00,  1.5264e+00,  1.3750e+00,  1.1598e+00,  1.1531e+00,
        1.1141e+00,  1.0403e+00,  9.9945e-01,  7.7199e-01,  6.9510e-01,
        5.7841e-01,  5.7347e-01,  5.0515e-01,  4.9448e-01,  4.3117e-01,
        3.9302e-01,  2.8671e-01,  2.8156e-01,  2.7193e-01,  2.0819e-01,
        1.2470e-01,  1.1865e-01,  1.1343e-01, -3.7308e-02, -5.9862e-02,
        -7.7117e-02, -1.2243e-01, -1.2354e-01, -2.1367e-01, -2.1852e-01,
        -2.1973e-01, -2.3444e-01, -2.4874e-01, -2.8249e-01, -3.0409e-01],
        [ 5.6472e+00,  5.5637e+00,  5.3926e+00,  4.8616e+00,  4.5172e+00,
        3.6336e+00,  3.5242e+00,  3.4385e+00,  3.3853e+00,  3.4170e+00,
        3.1216e+00,  3.2112e+00,  2.8267e+00,  2.6401e+00,  2.4389e+00,
        2.4203e+00,  2.5292e+00,  2.1249e+00,  2.1185e+00,  1.9485e+00,
        2.0025e+00,  2.0054e+00,  1.9556e+00,  1.7090e+00,  1.7862e+00,
        1.6869e+00,  1.6685e+00,  1.7540e+00,  1.7813e+00,  1.4229e+00,
        1.5010e+00,  1.8533e+00,  1.2905e+00,  1.1181e+00,  1.3787e+00,
        8.9487e-01,  9.4292e-01,  9.4067e-01,  1.2244e+00,  8.5835e-01,
        1.4543e+00,  6.1043e-01,  1.2296e+00,  1.0473e+00,  5.6585e-01,
        4.9595e-01,  1.1469e+00,  1.1285e+00,  1.0701e+00,  7.7133e-01],
        [-4.9961e-02, -5.1975e-02, -4.4834e-02, -5.3990e-02, -5.8750e-02,
        -4.7214e-02, -5.3074e-02, -6.4976e-02, -8.0358e-02,  3.1157e-02,
        -7.7794e-02,  8.2794e-02,  1.7973e-02, -3.4946e-02, -1.0929e-01,
        -3.8608e-02,  1.3700e-01, -1.4152e-01,  8.9386e-02, -9.0429e-02,
        1.2663e-02,  9.6162e-02,  8.6457e-02,  6.5582e-02,  2.3130e-01,
        2.4943e-01,  2.3514e-01,  3.9701e-01,  4.3657e-01,  1.2509e-01,
        2.4869e-01,  7.2496e-01,  1.4285e-01, -3.3664e-02,  3.1461e-01,
        -1.1863e-01, -5.9666e-02, -5.6370e-02,  4.1020e-01,  4.5440e-02,
        6.8871e-01, -1.6093e-01,  5.0487e-01,  4.0873e-01, -1.0618e-01,
        -1.8473e-01,  5.3380e-01,  5.2959e-01,  5.0395e-01,  2.1244e-01],
        [-3.2101e-02, -3.6300e-02, -3.3858e-02, -2.4971e-02, -1.6768e-02,
        -5.8664e-02, -4.1183e-02,  1.0284e-02,  2.6104e-02, -7.0187e-02,
        -6.9112e-02, -7.2335e-02, -1.2507e-01, -4.2161e-02,  1.2626e-02,
        -6.6088e-02,  3.8441e-01, -1.4763e-01, -5.5340e-04, -1.7195e-01,
        -8.7183e-02, -1.7498e-01, -1.6696e-01,  3.0707e-01,  8.4165e-01,
        5.2613e-01,  6.4380e-01,  7.6295e-02,  1.2503e-01,  5.3716e-01,
        4.0912e-01,  2.4292e-01,  7.4996e-01, -1.7956e-01,  6.5864e-01,
        -1.0631e-01,  1.0585e-02,  7.0074e-01,  2.8872e-01, -1.8268e-01,
        7.4357e-02,  6.2748e-01,  1.1194e-01,  5.0131e-01, -2.0614e-01,
        -2.0135e-01,  5.2994e-01,  7.1803e-01,  8.1676e-01,  2.6333e-01]],

        [[ 3.9476e+00,  3.5190e+00,  3.2789e+00,  3.2393e+00,  3.1768e+00,
        2.8118e+00,  2.3424e+00,  2.2837e+00,  2.1788e+00,  1.7707e+00,
        1.4570e+00,  1.4274e+00,  1.3485e+00,  1.3113e+00,  1.2472e+00,
        1.0257e+00,  1.0215e+00,  9.7249e-01,  9.2640e-01,  8.4086e-01,
        7.4639e-01,  7.4361e-01,  6.2181e-01,  5.5458e-01,  4.3750e-01,
        3.9959e-01,  2.9692e-01,  2.9255e-01,  9.7616e-02,  8.7825e-02,
        7.6129e-02, -2.1220e-02, -3.3262e-02, -3.4777e-02, -5.9862e-02,
        -9.8979e-02, -1.3130e-01, -1.3577e-01, -1.8037e-01, -2.5124e-01,
        -2.6834e-01, -2.7732e-01, -2.9094e-01, -3.2280e-01, -3.4737e-01,
        -3.5222e-01, -3.5989e-01, -3.6691e-01, -3.8397e-01, -3.8972e-01],
        [ 3.9523e+00,  3.5244e+00,  3.2825e+00,  3.2509e+00,  3.1842e+00,
        2.8222e+00,  2.3533e+00,  2.3030e+00,  2.1803e+00,  1.7728e+00,
        1.4599e+00,  1.4378e+00,  1.3503e+00,  1.3226e+00,  1.2825e+00,
        1.0284e+00,  1.0293e+00,  1.2987e+00,  1.0049e+00,  1.1704e+00,
        9.9745e-01,  9.9125e-01,  9.2390e-01,  5.6389e-01,  4.5772e-01,
        4.5118e-01,  3.1207e-01,  3.1623e-01,  1.3766e-01,  1.5711e-01,
        7.6156e-02, -4.7399e-03, -1.0184e-02, -2.2191e-02,  2.8498e-01,
        1.8012e-02, -1.0717e-01, -9.2723e-02, -6.6659e-02, -1.4985e-01,
        -2.6303e-01, -2.5872e-01,  2.2939e-02, -3.0410e-01, -2.0208e-01,
        -3.1752e-01, -2.2377e-01, -3.4668e-01, -3.2585e-01, -3.5870e-01],
        [-7.5982e-02, -6.8108e-02, -8.8068e-02, -1.9950e-02, -5.0713e-02,
        -2.8740e-02, -2.5627e-02,  2.4912e-02, -1.2139e-01, -1.1279e-01,
        -1.0345e-01, -3.2219e-02, -2.2156e-01, -2.2148e-02,  9.1747e-02,
        -1.1846e-01, -5.8404e-02,  6.7862e-01,  2.2524e-01,  6.8301e-01,
        5.6381e-01,  5.5831e-01,  6.4200e-01, -6.1700e-02, -3.5376e-01,
        1.3844e-01, -3.1366e-01,  2.0151e-02,  8.4057e-02,  1.8349e-01,
        -1.6534e-01, -5.8037e-02, -3.3454e-01, -1.0473e-01,  6.9839e-01,
        2.9940e-01, -1.7936e-02,  7.8380e-02,  3.1331e-01,  2.5325e-01,
        -2.7576e-01, -1.0949e-01,  6.4786e-01, -2.0379e-01,  3.4938e-01,
        6.6008e-03,  3.2851e-01, -1.9995e-01,  1.7158e-01, -2.6359e-02],
        [ 7.1875e-02,  7.0312e-02,  7.8208e-02,  2.5790e-01,  2.4638e-01,
        4.5311e-02,  2.9931e-01,  2.2763e-01,  4.3748e-02,  9.6176e-02,
        2.2303e-01,  1.2597e-01, -6.2995e-02, -6.9387e-01,  2.3798e-01,
        -2.2754e-02,  2.8037e-01, -5.1877e-01, -8.5452e-01, -7.5735e-01,
        -7.8352e-01, -8.0071e-01, -6.1048e-01, -7.2258e-01, -8.0070e-01,
        -1.5040e-01,  2.8367e-03,  3.0029e-01, -1.1360e+00, -1.8860e-02,
        1.8367e-01, -2.5459e-01, -3.1201e-01, -4.0567e-01, -3.3457e-01,
        -1.0039e-01, -1.2557e+00, -4.6965e-01, -6.7327e-01, -1.2008e+00,
        -4.0596e-01, -6.5646e-01, -4.2087e-02, -5.4962e-01, -4.9620e-01,
        -9.6233e-01, -1.9670e-01, -6.6125e-02, -6.1272e-01, -7.3099e-01]],

        [[ 4.5907e+00,  3.5774e+00,  3.0430e+00,  2.9453e+00,  2.5846e+00,
        2.5264e+00,  2.4417e+00,  2.2963e+00,  2.2059e+00,  1.7931e+00,
        1.2900e+00,  1.2852e+00,  1.2164e+00,  1.1948e+00,  1.0396e+00,
        1.0368e+00,  1.0166e+00,  9.8351e-01,  9.7544e-01,  7.4732e-01,
        7.1536e-01,  6.6592e-01,  6.3432e-01,  5.9201e-01,  5.8278e-01,
        5.7292e-01,  5.4952e-01,  5.4217e-01,  5.0338e-01,  4.9567e-01,
        4.7098e-01,  4.4504e-01,  4.3497e-01,  3.7506e-01,  3.6630e-01,
        2.8671e-01,  2.7193e-01,  2.6447e-01,  2.5920e-01,  2.4019e-01,
        2.1530e-01,  2.0262e-01,  1.4855e-01,  1.4518e-01,  1.1343e-01,
        1.1081e-01,  1.0203e-01,  8.6930e-02,  7.2502e-02,  5.9705e-02],
        [ 5.0602e+00,  4.0415e+00,  3.4361e+00,  3.4130e+00,  3.0537e+00,
        2.9869e+00,  2.9179e+00,  2.7305e+00,  2.6700e+00,  2.3124e+00,
        2.6968e+00,  2.6646e+00,  1.2206e+00,  1.6419e+00,  1.5210e+00,
        2.3212e+00,  1.2885e+00,  2.7482e+00,  1.4048e+00,  1.2766e+00,
        7.3007e-01,  7.6287e-01,  2.1247e+00,  9.1355e-01,  1.1256e+00,
        6.5424e-01,  8.3462e-01,  1.1267e+00,  9.6772e-01,  1.0293e+00,
        4.8248e-01,  4.5294e-01,  4.8259e-01,  4.9714e-01,  1.8010e+00,
        3.0653e-01,  8.9514e-01,  2.7319e-01,  3.1891e-01,  1.9589e+00,
        2.5452e-01,  3.1638e-01,  1.6327e-01,  3.1663e-01,  1.5412e+00,
        1.4247e+00,  6.9231e-01,  3.3167e-01,  3.9914e-01,  1.1857e+00],
        [ 1.6225e-01,  1.6921e-01,  2.6277e-01,  1.6444e-01,  1.6280e-01,
        1.7378e-01,  1.5364e-01,  2.0802e-01,  1.6921e-01,  9.9440e-02,
        -8.7599e-01, -8.4779e-01,  1.2911e+00,  1.9154e-01,  1.4705e-01,
        -7.4910e-01,  4.3728e-01, -1.2418e+00,  2.1517e-01,  8.7904e-02,
        1.3805e+00,  1.6509e+00, -9.6187e-01,  3.6532e-01,  7.1607e-02,
        8.0661e-01,  4.1988e-01,  1.9603e-02,  1.6884e-01,  8.3143e-02,
        1.0566e+00,  1.1196e+00,  1.5070e+00,  1.7050e+00, -9.0456e-01,
        1.3794e+00, -2.4710e-02,  1.1304e+00,  8.7528e-01, -1.1950e+00,
        1.4682e+00,  1.6830e+00,  1.0848e+00,  1.8013e+00, -8.9723e-01,
        -7.7931e-01,  1.5575e-02,  4.8818e-01,  3.6202e-01, -5.8228e-01],
        [-2.1986e-01, -2.2523e-01, -9.0360e-02, -2.4173e-01, -1.7083e-01,
        -2.3001e-01, -2.8754e-01, -2.0218e-01, -1.9837e-01, -1.2396e-01,
        1.2576e+00,  2.2293e-01, -1.1937e-02, -2.3167e-01, -1.7289e-01,
        5.2411e-01,  4.5995e-01,  4.3690e-01,  3.2928e-01, -2.2864e-01,
        -3.6967e-01,  6.0528e-02,  1.3142e+00,  1.3249e-01, -2.7298e-01,
        2.3661e-01,  1.0076e-01, -3.7415e-01, -3.7631e-02,  6.4911e-01,
        -8.9190e-03,  6.1904e-01, -5.4447e-01, -1.0609e-01,  1.4284e-01,
        -4.7894e-01, -1.8088e-01,  6.6016e-01,  9.9853e-01,  6.6092e-01,
        -3.9555e-01,  2.2810e-01,  7.5284e-01,  3.7665e-01,  1.0564e+00,
        -1.7981e-01,  6.3701e-01,  7.0741e-01,  8.1024e-01,  7.6317e-01]],

        [[ 3.1009e+00,  2.7266e+00,  2.4383e+00,  2.4245e+00,  2.2170e+00,
        2.0273e+00,  1.9542e+00,  1.8658e+00,  1.7410e+00,  1.6288e+00,
        1.6211e+00,  1.4227e+00,  1.1823e+00,  9.8424e-01,  9.7912e-01,
        9.4782e-01,  8.8293e-01,  8.6500e-01,  8.2900e-01,  8.0396e-01,
        6.3949e-01,  6.3432e-01,  6.3017e-01,  5.9686e-01,  5.9093e-01,
        5.6574e-01,  5.5962e-01,  5.2440e-01,  4.7524e-01,  4.3687e-01,
        4.2289e-01,  4.1389e-01,  4.0612e-01,  3.3954e-01,  3.3675e-01,
        2.8082e-01,  2.2937e-01,  2.2704e-01,  1.8004e-01,  1.6442e-01,
        1.2557e-01,  1.2470e-01,  1.1865e-01,  1.1169e-01,  1.0994e-01,
        9.3177e-02,  6.4294e-02,  1.3579e-02,  1.9512e-03, -5.6240e-02],
        [ 3.2675e+00,  2.8280e+00,  2.6170e+00,  2.5558e+00,  2.3648e+00,
        2.1113e+00,  2.0319e+00,  1.9873e+00,  1.9095e+00,  1.6556e+00,
        1.6908e+00,  1.5681e+00,  1.2057e+00,  1.2099e+00,  1.3350e+00,
        1.0173e+00,  9.4828e-01,  8.8847e-01,  1.3937e+00,  8.4475e-01,
        7.2671e-01,  7.5079e-01,  9.1116e-01,  7.5076e-01,  7.0071e-01,
        2.2228e+00,  1.7340e+00,  1.2972e+00,  1.7145e+00,  7.7951e-01,
        1.6356e+00,  1.3809e+00,  1.8099e+00,  1.5072e+00,  4.3526e-01,
        2.9415e-01,  3.9627e-01,  1.5500e+00,  2.0036e+00,  2.7436e-01,
        2.6517e-01,  5.5860e-01,  2.4453e-01,  2.4598e-01,  8.4765e-01,
        1.1793e-01,  1.1249e+00,  1.1757e+00,  3.6908e-01,  1.3255e+00],
        [-5.3100e-01, -6.6650e-01, -5.0847e-01, -6.0076e-01, -5.6725e-01,
        -7.0898e-01, -7.2546e-01, -6.2164e-01, -5.2770e-01, -8.9319e-01,
        -7.4652e-01, -5.7274e-01, -9.1095e-01, -4.2845e-01, -2.2978e-01,
        -7.4707e-01, -7.5879e-01, -9.1461e-01,  3.9945e-02, -8.4320e-01,
        -7.0623e-01, -6.3702e-01, -3.4148e-01, -5.5938e-01, -6.4709e-01,
        1.2166e+00,  7.1819e-01,  2.8275e-01,  7.8612e-01, -2.5138e-01,
        7.5829e-01,  4.9699e-01,  9.5696e-01,  7.1086e-01, -6.7309e-01,
        -9.9848e-01, -5.3869e-01,  8.7365e-01,  1.3856e+00, -6.4672e-01,
        -5.9527e-01, -1.2412e-01, -6.1193e-01, -6.0680e-01,  2.4173e-01,
        -9.3933e-01,  5.9807e-01,  7.0464e-01, -2.2062e-01,  9.3371e-01],
        [ 1.8448e-01,  1.6963e-01,  2.1133e-01,  2.0860e-01,  1.7608e-01,
        2.1485e-01,  2.6114e-01,  2.3994e-01,  1.4502e-01,  1.9492e-01,
        1.3027e-01,  1.7452e-01,  2.0996e-01,  2.6035e-01, -6.2942e-01,
        2.2979e-01,  1.6709e-01,  3.2960e-01, -4.6848e-01,  3.2657e-01,
        2.1787e-01,  1.5097e-01, -1.5907e+00, -1.4102e+00,  7.9291e-02,
        -1.0000e-01, -3.7638e-01, -1.2456e+00,  6.7433e-01, -9.8870e-01,
        -2.8760e-01, -5.7160e-01,  4.2189e-01,  1.8574e-01,  4.0225e-01,
        -1.6618e+00,  2.7881e-01,  5.4825e-01,  2.7687e-01, -1.0495e+00,
        4.4074e-01, -7.8244e-01,  2.8957e-01, -1.3689e+00, -9.1155e-01,
        4.5595e-02, -6.0599e-01,  1.9853e-01, -1.3889e+00, -3.1358e-01]]])

pf_mask = torch.FloatTensor([[[True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True]],

        [[True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True]],

        [[True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True]],

        [[True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True]],

        [[True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True]]])

sv_points = torch.FloatTensor([[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

        [[ 0.0522,  0.0540,  0.0000,  0.0000,  0.0000],
        [ 0.0266,  0.0390,  0.0000,  0.0000,  0.0000]],

        [[ 0.0849,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0720,  0.0000,  0.0000,  0.0000,  0.0000]],

        [[-0.1591,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.2157,  0.0000,  0.0000,  0.0000,  0.0000]],

        [[ 0.6419,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.2050,  0.0000,  0.0000,  0.0000,  0.0000]]])

sv_features = torch.FloatTensor([[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

        [[ 4.1350,  4.8190,  0.0000,  0.0000,  0.0000],
        [ 0.0522,  0.0540,  0.0000,  0.0000,  0.0000],
        [ 0.0266,  0.0390,  0.0000,  0.0000,  0.0000]],

        [[ 4.3174,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0849,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0720,  0.0000,  0.0000,  0.0000,  0.0000]],

        [[ 4.0320,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.1591,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.2157,  0.0000,  0.0000,  0.0000,  0.0000]],

        [[ 3.5040,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.6419,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.2050,  0.0000,  0.0000,  0.0000,  0.0000]]])

sv_mask = torch.FloatTensor([[[False, False, False, False, False]],

        [[ True,  True, False, False, False]],

        [[ True, False, False, False, False]],

        [[ True, False, False, False, False]],

        [[ True, False, False, False, False]]])

import onnxruntime
ort_session = onnxruntime.InferenceSession(os.path.join(model_outpath, "model.onnx"))

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


ort_inputs = ort_session.get_inputs()[0]
print("ort_inputs", ort_inputs)
ort_inputs = ort_session.get_inputs()[1]
print("ort_inputs", ort_inputs)
ort_inputs = ort_session.get_inputs()[2]
print("ort_inputs", ort_inputs)
ort_inputs = ort_session.get_inputs()[3]
print("ort_inputs", ort_inputs)
ort_inputs = ort_session.get_inputs()[4]
print("ort_inputs", ort_inputs)
ort_inputs = ort_session.get_inputs()[5]
print("ort_inputs", ort_inputs)

ort_inputs = {
    ort_session.get_inputs()[0].name: to_numpy(pf_points),
    ort_session.get_inputs()[1].name: to_numpy(pf_features),
    ort_session.get_inputs()[2].name: to_numpy(pf_mask),
    ort_session.get_inputs()[3].name: to_numpy(sv_points),
    ort_session.get_inputs()[4].name: to_numpy(sv_features),
    ort_session.get_inputs()[5].name: to_numpy(sv_mask),
}

ort_outs = ort_session.run(None, ort_inputs)
print("ort_outs", ort_outs)
print("len(ort_outs)", len(ort_outs))
print("ort_outs[0]", ort_outs[0])
print("type(ort_outs[0])", type(ort_outs[0]))
print("ort_outs[0].shape", ort_outs[0].shape)
print("ort_outs[0].tolist()", ort_outs[0].tolist())
"""


