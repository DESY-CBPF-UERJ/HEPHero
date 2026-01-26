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
numpy_random = np.random.RandomState(16)
sys.path.append("..")
from custom_opts.ranger import Ranger
import tools


#==================================================================================================
class build_NN(nn.Module):
    # Constructor
    def __init__(self, parameters, variables, n_classes, stat_values, device):
        super(build_NN, self).__init__()

        n_var = len(variables)

        if device == "cuda":
            self.mean = torch.tensor(stat_values["mean"], dtype=torch.float32).to("cuda")
            self.std = torch.tensor(stat_values["std"], dtype=torch.float32).to("cuda")
        else:
            self.mean = torch.tensor(stat_values["mean"], dtype=torch.float32)
            self.std = torch.tensor(stat_values["std"], dtype=torch.float32)

        self.apply_pca = False
        if "eigenvectors_pca" in stat_values:
            self.apply_pca = True
            if device == "cuda":
                self.eigenvectors_pca = torch.tensor(stat_values["eigenvectors_pca"], dtype=torch.float32).to("cuda")
                self.mean_pca = torch.tensor(stat_values["mean_pca"], dtype=torch.float32).to("cuda")
                self.std_pca = torch.tensor(stat_values["std_pca"], dtype=torch.float32).to("cuda")
            else:
                self.eigenvectors_pca = torch.tensor(stat_values["eigenvectors_pca"], dtype=torch.float32)
                self.mean_pca = torch.tensor(stat_values["mean_pca"], dtype=torch.float32)
                self.std_pca = torch.tensor(stat_values["std_pca"], dtype=torch.float32)

        if parameters[7][1] == 'relu':
            self.activation_hidden = nn.ReLU()
        elif parameters[7][1] == 'tanh':
            self.activation_hidden = nn.Tanh()
        elif parameters[7][1] == 'elu':
            self.activation_hidden = nn.ELU()
        elif parameters[7][1] == 'gelu':
            self.activation_hidden = nn.GELU()
        elif parameters[7][1] == 'selu':
            self.activation_hidden = nn.SELU()
        else:
            print("Error: hidden activation function not supported!")

        self.apply_BatchNorm = False
        if parameters[7][2]:
            self.apply_BatchNorm = True
            self.BatchNorm = nn.BatchNorm1d(parameters[7][0][0])

        self.apply_Dropout = False
        if parameters[7][3] is not None:
            self.apply_Dropout = True
            self.Dropout = nn.Dropout(p=parameters[7][3])

        if parameters[4] == 'cce':
            self.activation_last = nn.Softmax(dim=1)
            n_output = n_classes
        elif parameters[4] == 'bce' and n_classes == 2:
            self.activation_last = nn.Sigmoid()
            n_output = 1
        else:
            print("Error: last activation function or number of classes is not supported!")

        self.hidden = nn.ModuleList()
        for i in range(len(parameters[7][0])):
            if i == 0:
                self.hidden.append(nn.Linear(n_var, parameters[7][0][i]))
            if i > 0:
                self.hidden.append(nn.Linear(parameters[7][0][i-1], parameters[7][0][i]))
        self.hidden.append(nn.Linear(parameters[7][0][-1], n_output))

    def stat_device(self, dev):
        self.mean = self.mean.to(dev)
        self.std = self.std.to(dev)
        if self.apply_pca:
            self.eigenvectors_pca = self.eigenvectors_pca.to(dev)
            self.mean_pca = self.mean_pca.to(dev)
            self.std_pca = self.std_pca.to(dev)

    # Prediction
    def forward(self, x):

        x = (x - self.mean) / self.std

        if self.apply_pca:
            x = torch.matmul(x, self.eigenvectors_pca)
            x = (x - self.mean_pca) / self.std_pca

        N_layers = len(self.hidden)
        for i, layer in enumerate(self.hidden):
            if i < N_layers-1:
                x = layer(x)
                if self.apply_BatchNorm:
                    x = self.BatchNorm(x)
                x = self.activation_hidden(x)
                if self.apply_Dropout:
                    x = self.Dropout(x)
            else:
                x = layer(x)
                x = self.activation_last(x)

        return x


#==================================================================================================
def model_parameters_NN(param_dict):
    num_layers = param_dict["num_layers"]
    num_nodes = param_dict["num_nodes"]
    activation_func = param_dict["activation_func"]
    batch_norm = param_dict["batch_norm"]
    dropout = param_dict["dropout"]

    model_parameters = []
    for i_num_layers in num_layers:
        for i_num_nodes in num_nodes:
            for i_activation_func in activation_func:
                for i_batch_norm in batch_norm:
                    for i_dropout in dropout:
                        model_parameters.append([[i_num_nodes for i in range(i_num_layers)]] + [i_activation_func] + [i_batch_norm] + [i_dropout])

    return model_parameters


#==================================================================================================
def features_stat_NN(train_data, test_data, variables, var_names, class_names, class_labels, class_colors, plots_outpath, load_it=None):

    train_data = pd.DataFrame.from_dict(train_data)
    test_data = pd.DataFrame.from_dict(test_data)

    mean = []
    std = []
    dim = len(variables)
    for i in range(len(variables)):
        weighted_stats = DescrStatsW(train_data[variables[i]], weights=train_data["mvaWeight"], ddof=0)
        mean.append(weighted_stats.mean)
        std.append(weighted_stats.std)
    np.set_printoptions(legacy='1.25')
    if load_it is None:
        print("mean: " + str(mean))
        print("std: " + str(std))
        print("dim: " + str(dim))
    stat_values={"mean": mean, "std": std, "dim": dim}


    for ivar in range(len(variables)):

        fig1 = plt.figure(figsize=(9,5))
        gs1 = gs.GridSpec(1, 1)
        #==================================================
        ax1 = plt.subplot(gs1[0])
        #==================================================
        var = variables[ivar]
        bins = np.linspace(mean[ivar]-5*std[ivar],mean[ivar]+5*std[ivar],101)
        for ikey in range(len(class_names)):
            tools.step_plot( ax1, var, train_data[train_data["class"] == ikey], label=class_labels[ikey]+" (train)", color=class_colors[ikey], weight="mvaWeight", bins=bins, error=True )
            tools.step_plot( ax1, var, test_data[test_data["class"] == ikey], label=class_labels[ikey]+" (test)", color=class_colors[ikey], weight="mvaWeight", bins=bins, error=True, linestyle='dotted' )
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
        if load_it is None:
            plt.savefig(os.path.join(plots_outpath, var + '.png'), dpi=400)
            plt.savefig(os.path.join(plots_outpath, var + '.pdf'))
        else:
            plt.savefig(os.path.join(plots_outpath, var +"_"+ str(load_it) + '.png'), dpi=400)
            plt.savefig(os.path.join(plots_outpath, var +"_"+ str(load_it) + '.pdf'))
        plt.close()

    return stat_values


#==================================================================================================
def update_NN(model, criterion, parameters, batch_data, device):

    if parameters[3] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters[6], eps=1e-07)
        # lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
    elif parameters[3] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=parameters[6])
        # lr=?, momentum=0, dampening=0, weight_decay=0, nesterov=False
    elif parameters[3] == "ranger":
        optimizer = Ranger(model.parameters(), lr=parameters[6])
    #print(optimizer.state_dict())

    data_x_b, data_y_b, data_w_b = batch_data
    del batch_data

    if device == "cuda":
        w = torch.FloatTensor(data_w_b).view(-1,1).to("cuda")
        x = torch.FloatTensor(data_x_b).to("cuda")
        y = torch.IntTensor(data_y_b).view(-1,1).to("cuda")
    else:
        w = torch.FloatTensor(data_w_b).view(-1,1)
        x = torch.FloatTensor(data_x_b)
        y = torch.IntTensor(data_y_b).view(-1,1)

    x.requires_grad=True
    yhat = model(x)
    #print("yhat type ", yhat.dtype)

    loss = criterion(y, yhat, w, device=device)
    #print("Loss = ", loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return model


#==================================================================================================
def process_data_NN(scalar_var, variables):

    scalar_var = pd.DataFrame.from_dict(scalar_var)
    data_x = scalar_var[variables]
    data_x = data_x.values
    data_y = np.array(scalar_var['class']).ravel()
    data_w = np.array(scalar_var['mvaWeight']).ravel()

    input_data = [data_x, data_y, data_w]

    return input_data


#==================================================================================================
def evaluate_NN(input_data, model, i_eval, eval_step_size, criterion, parameters, device, mode):

    data_x, data_y, data_w = input_data

    n_eval_steps = int(len(data_w)/eval_step_size) + 1
    last_eval_step = len(data_w)%eval_step_size

    if i_eval < n_eval_steps-1:
        eval_data_x = data_x[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
        eval_data_y = data_y[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
        eval_data_w = data_w[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
    elif last_eval_step > 0:
        eval_data_x = data_x[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_step]
        eval_data_y = data_y[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_step]
        eval_data_w = data_w[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_step]
    else:
        return None

    if device == "cuda":
        eval_data_yhat = model(torch.FloatTensor(eval_data_x).to("cuda"))
        eval_data_yhat = eval_data_yhat.cpu()
    else:
        eval_data_yhat = model(torch.FloatTensor(eval_data_x))

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
def feature_score_NN(input_data, model, min_loss, eval_step_size, criterion, parameters, variables, var_names, device):

    n_eval_steps = int(len(input_data[-1])/eval_step_size) + 1
    data_w_sum = input_data[-1].sum()

    features_score = []
    features_score_unc = []
    features_names = []
    for ivar in tqdm(range(len(variables))):
        losses = []
        for irep in range(30):

            data_x, data_y, data_w = input_data
            data_x_shuffled = data_x.copy()
            numpy_random.shuffle(data_x_shuffled[:,ivar])
            shuffled_data = [data_x_shuffled, data_y, data_w]

            data_loss_i = 0
            for i_eval in range(n_eval_steps):
                i_eval_output = evaluate_NN(shuffled_data, model, i_eval, eval_step_size, criterion, parameters, device, mode="metric")
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
        features_names.append(var_names[ivar])

    feature_score_info = features_score, features_score_unc, features_names

    return feature_score_info


#==================================================================================================
def save_NN(model, model_outpath, dim, device):

    #torch.save(model, os.path.join(model_outpath, "model.pt"))
    #model_scripted = torch.jit.script(model) # Export to TorchScript
    #model_scripted.save(os.path.join(model_outpath, "model_scripted.pt"))
    #To evaluate model_state_dict.pt
    #model = TheModelClass(*args, **kwargs)
    #model.load_state_dict(torch.load(PATH, weights_only=True))
    #model.eval()

    #ONNX
    input_names = ['features']
    output_names = ['output']
    dynamic_axes = {'features': {0: 'N'}, 'output': {0: 'N'}}
    input_shapes = {'features': (1, dim)}
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




















