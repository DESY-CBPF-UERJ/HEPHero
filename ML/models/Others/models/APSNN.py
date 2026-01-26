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
from madminer.utils import morphing as m
import tools
import random
import concurrent.futures
import multiprocessing
from itertools import repeat

class AffineConditioning(nn.Module):
    def __init__(self, input_dim, par_dim, output_dim, BatchNorm=False, Dropout=None, activation_func="relu"):
        super(AffineConditioning, self).__init__()
        self.apply_BatchNorm = BatchNorm
        if BatchNorm:
            self.BatchNorm = nn.BatchNorm1d(output_dim)

        self.apply_Dropout = False
        if Dropout is not None:
            self.apply_Dropout = True
            self.Dropout = nn.Dropout(p=Dropout)

        if activation_func == 'relu':
            self.activation_hidden = nn.ReLU()
        elif activation_func == 'tanh':
            self.activation_hidden = nn.Tanh()
        elif activation_func == 'elu':
            self.activation_hidden = nn.ELU()
        elif activation_func == 'gelu':
            self.activation_hidden = nn.GELU()
        elif activation_func == 'selu':
            self.activation_hidden = nn.SELU()
        else:
            print("Error: hidden activation function not supported!")
        
        self.linear_h = nn.Linear(input_dim, output_dim)
        self.linear_s = nn.Linear(par_dim, output_dim)
        self.linear_b = nn.Linear(par_dim, output_dim)
    
    def forward(self, x, par):
        x = self.linear_h(x) 
        if self.apply_BatchNorm:
            x = self.BatchNorm(x)
        x = self.activation_hidden(x)
        s = self.linear_s(par)
        b = self.linear_b(par)
        x = x*s+b
        if self.apply_Dropout:
            x = self.Dropout(x)
            
        return x

#==================================================================================================
class build_APSNN(nn.Module):
    # Constructor
    def __init__(self, parameters, variables, n_classes, stat_values, device):
        super(build_APSNN, self).__init__()

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

        self.par_dim = stat_values["par_dim"]
        self.input_dim = n_var - self.par_dim
        self.hidden = nn.ModuleList()
        for i in range(len(parameters[7][0])):
            if i == 0:
                self.hidden.append(AffineConditioning(self.input_dim, self.par_dim, parameters[7][0][i], BatchNorm=parameters[7][2], Dropout=parameters[7][3], activation_func=parameters[7][1]))
            if i > 0:
                self.hidden.append(AffineConditioning(parameters[7][0][i-1], self.par_dim, parameters[7][0][i], BatchNorm=parameters[7][2], Dropout=parameters[7][3], activation_func=parameters[7][1]))
        self.hidden.append(nn.Linear(parameters[7][0][-1], self.par_dim))

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

        par = x[:,self.input_dim:]
        x = x[:,:self.input_dim]

        N_layers = len(self.hidden)
        for i, layer in enumerate(self.hidden):
            if i < N_layers-1:
                x = layer(x, par) 
            else:
                x = layer(x)

        return x


#==================================================================================================
def model_parameters_APSNN(param_dict):
    affine_setups = param_dict["affine_setups"]
    activation_func = param_dict["activation_func"]
    batch_norm = param_dict["batch_norm"]
    dropout = param_dict["dropout"]
    parameter_max_power = param_dict["parameter_max_power"]
    max_overall_power = param_dict["max_overall_power"]
    basis = param_dict["basis"]

    model_parameters = []
    for i_affine_setups in affine_setups:
        for i_activation_func in activation_func:
            for i_batch_norm in batch_norm:
                for i_dropout in dropout:
                    model_parameters.append([i_affine_setups] + [i_activation_func] + [i_batch_norm] + [i_dropout] + [parameter_max_power] + [max_overall_power] + [basis])

    return model_parameters


#==================================================================================================
def features_stat_APSNN(train_data, test_data, variables, var_names, var_use, class_names, class_labels, class_colors, plots_outpath, parameters, load_it=None):

    train_data = pd.DataFrame.from_dict(train_data)
    test_data = pd.DataFrame.from_dict(test_data)

    par_idx = []
    par_dim = 0
    for i in range(len(variables)):
        if len(var_use[i]) > 2:
            sys.exit("Code does not support more than 2 signal classes!")
        if var_use[i] != "F":
            par_dim += 1
            par_idx.append(i)
            
    train_parameter_points = np.array([numpy_random.normal(loc=0.0, scale=1.0, size=par_dim) for ip in range(500)])
    test_parameter_points = np.array([numpy_random.normal(loc=0.0, scale=1.0, size=par_dim) for ip in range(500)])
    train_elements = numpy_random.choice(500, size=len(train_data))
    test_elements = numpy_random.choice(500, size=len(test_data))
    for i in range(par_dim):
        idx = par_idx[i]
        train_data[variables[idx]] = np.array([train_parameter_points[train_elements[ie],i] for ie in range(len(train_data))])
        test_data[variables[idx]] = np.array([test_parameter_points[test_elements[ie],i] for ie in range(len(test_data))])

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
        print("par_dim: " + str(par_dim))
        print("par_idx: " + str(par_idx))
    stat_values={"mean": mean, "std": std, "dim": dim, "par_dim": par_dim, "par_idx": par_idx}

    if parameters is not None:
        morpher = m.PhysicsMorpher(parameter_max_power=parameters[7][4])
        components = morpher.find_components(max_overall_power=parameters[7][5])
        if len(components) == len(parameters[7][6]):
            morpher.set_basis(basis_numpy=np.array(parameters[7][6]))
        else:
            sys.exit("It was expected a basis with " + str(len(components)) + " components, but " + str(len(parameters[7][6])) + " was provided!")
        morpher.calculate_morphing_matrix()        
        stat_values["morpher"] = morpher 

    for ivar in range(len(variables)):
        fig1 = plt.figure(figsize=(9,5))
        gs1 = gs.GridSpec(1, 1)
        #==================================================
        ax1 = plt.subplot(gs1[0])
        #==================================================
        var = variables[ivar]
        bins = np.linspace(mean[ivar]-5*std[ivar],mean[ivar]+5*std[ivar],51)
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
def update_APSNN(model, criterion, parameters, batch_data, var_use, device):

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
        y = torch.IntTensor(data_y_b).to("cuda")
    else:
        w = torch.FloatTensor(data_w_b).view(-1,1)
        x = torch.FloatTensor(data_x_b)
        y = torch.IntTensor(data_y_b)

    #print("shapes", x.shape, w.shape, y.shape)

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
def process_data_APSNN(scalar_var, variables, var_use, vector_var, stat_values, device):

    basis_indexes = [200, 201, 202, 203, 204, 207, 208, 209, 210, 211, 217, 218, 219, 220, 223, 224, 225, 226, 227, 233, 234, 235, 238, 239, 240, 241, 242, 248, 249, 252, 253, 254, 255, 256, 262, 265, 266, 267, 268, 269, 298, 299, 300, 301, 302, 308, 309, 310, 311, 317, 318, 319, 325, 326, 332]

    basis_indexes = [0, 5, 1, 7, 12, 3]
    
    #for i in basis_indexes:
    #    print(vector_var["eftWeights"][0,i])

    scalar_var = pd.DataFrame.from_dict(scalar_var)  

    n_points = 150 # transform it in a parameter
    #------------------------------------------------------
    # Produce random values for EFT parameters
    par_dim = stat_values["par_dim"]
    parameter_points = np.array([numpy_random.normal(loc=0.0, scale=1.0, size=par_dim) for ip in range(n_points)])
    elements = numpy_random.choice(n_points, size=len(scalar_var))
    #print(elements[0:100])
    data_x_param = []
    for i in range(par_dim):
        idx = stat_values["par_idx"][i]
        scalar_var[variables[idx]] = np.array([parameter_points[elements[ie],i] for ie in range(len(scalar_var))])
        data_x_param.append(scalar_var[variables[idx]].copy())
    data_x = scalar_var[variables]
    data_x = data_x.values
    #print("data_x", data_x.shape)
    data_x_param = np.array(data_x_param).T
    #print("data_x_param", data_x_param.shape)
    
    #------------------------------------------------------
    eft_weights = np.array([vector_var["eftWeights"][:,i] for i in basis_indexes]).T
    data_w_original = np.array(scalar_var['mvaWeight']).ravel()
    data_w_eft = eft_weights*data_w_original[:,None]

    morpher = stat_values["morpher"]      
    morphing_weights = morpher.calculate_morphing_weights(theta=data_x_param)

    ind_points = np.argsort(elements)
    element_points = elements[ind_points]
    idx_unique = np.unique(element_points, return_index=True)[1]
    #------------------------------------------------------
    morphing_weight_points = morphing_weights[ind_points]
    morphing_weight_points = morphing_weight_points[idx_unique]
    #------------------------------------------------------
    
    data_w = np.array([np.dot(morphing_weights[i,:],data_w_eft[i,:]) for i in range(len(data_w_eft))])
    #data_w2 = np.array([np.dot(morphing_weights[i,:],data_w_eft[i,:]*data_w_eft[i,:]) for i in range(len(data_w_eft))])
    #print("data_w", data_w.shape)
    #print("data_w2", data_w2.shape)
    #print("data_w_b", data_w_b[0], data_w_b[100], data_w_b[500])

    morphing_weights_grad = morpher.calculate_morphing_weight_gradient(theta=data_x_param, device=device)
    #print("morphing_weights_grad", morphing_weights_grad.shape)

    #------------------------------------------------------
    morphing_weights_grad_points = morphing_weights_grad[ind_points]
    morphing_weights_grad_points = morphing_weights_grad_points[idx_unique]
    #------------------------------------------------------
    
    data_w_grad = np.array([np.dot(morphing_weights_grad[i,:],data_w_eft[i,:]) for i in range(len(data_w_eft))])       

    #------------------------------------------------------
    data_xsec_points = np.array([np.array([np.dot(morphing_weight_points[ip,:],data_w_eft[ie,:]) for ie in range(len(data_w_eft))]).sum() for ip in range(n_points)])  
    #print("data_xsec_points", data_xsec_points.shape)

    data_xsec = np.array([data_xsec_points[elements[ie]] for ie in range(len(data_w_eft))])
    #data_xsec = data_w.sum() # xsec  # WRONG
    
    #print("data_xsec", data_xsec)
    #data_w_sum_unc = np.sqrt(np.power(data_w,2).sum()) # xsec unc.
    #print("data_xsec_unc", data_w_sum_unc)
    #data_w_sum_unc = np.sqrt(np.maximum(data_w2.sum(),0.)) # xsec unc.
    #print("data_xsec_unc", data_w_sum_unc)

    cpu_count = multiprocessing.cpu_count()
    if cpu_count <= 2:
        cpu_count = 1
    else:
        cpu_count -= 2

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        data_xsec_grad_points = np.array(list(executor.map(get_data_xsec_grad_points, range(n_points), repeat(morphing_weights_grad_points), repeat(data_w_eft))))
    
    #data_xsec_grad_points = np.array([np.array([np.dot(morphing_weights_grad_points[ip,:],data_w_eft[ie,:]) for ie in range(len(data_w_eft))]).sum() for ip in range(n_points)]) # time consuming

    #data_xsec_grad = data_w_grad.sum(axis=0) # xsec_grad  # WRONG
    data_xsec_grad = np.array([data_xsec_grad_points[elements[ie]] for ie in range(len(data_w_eft))])

    #xsec_shape ()
    #xsec_grad_shape (9,)
    #data_w_shape (49999,)
    #data_w_grad_shape (49999, 9)

    #xsec_shape (49999,)
    #xsec_grad_shape (49999, 9)
    #data_w_shape (49999,)
    #data_w_grad_shape (49999, 9)

    """
    score1 = data_w_grad[:, :] / data_w[:, np.newaxis] 
    score = score1 - data_xsec_grad[np.newaxis, :] / data_xsec[np.newaxis, np.newaxis]
    score2 = score1 - score
    """

    score1 = data_w_grad[:, :] / data_w[:, np.newaxis]
    score2 = data_xsec_grad[:, :] / data_xsec[:, np.newaxis]
    score = data_w_grad[:, :] / data_w[:, np.newaxis] - data_xsec_grad[:, :] / data_xsec[:, np.newaxis]
    print("score_shape", score.shape)

    """
    data_w_2D = data_w_grad/score1
    data_xsec_2D = data_xsec_grad[np.newaxis, :]/score2
    data_xsec_grad_2D = score2*data_xsec_2D

    score4 = np.array([[data_w_2D[:,i], data_w_grad[:,i], data_xsec_2D[:,i], data_xsec_grad_2D[:,i]] for i in range(par_dim)])
    score4 = np.moveaxis(score4.T, 1, 2)
    #print("score4_shape", score4.shape)
    """

    print("xsec_shape", data_xsec.shape)
    print("xsec_grad_shape", data_xsec_grad.shape)
    print("data_w_shape", data_w.shape)
    print("data_w_grad_shape", data_w_grad.shape)

    print("xsec", np.mean(data_xsec), np.std(data_xsec))
    print("xsec_grad", np.mean(data_xsec_grad[0]), np.std(data_xsec_grad[0]))
    print("data_w", np.mean(data_w), np.std(data_w))
    print("data_w_grad", np.mean(data_w_grad[:,0]), np.std(data_w_grad[:,0]))
    print("score_info", np.mean(score[:,0]), np.std(score[:,0]))
    print("score1_info", np.mean(score1[:,0]), np.std(score1[:,0]))
    print("score2_info", np.mean(score2[:,0]), np.std(score2[:,0]))
    
    #data_y = data_w_grad
    data_y = score
    #print("data_y", data_y.shape)
    #print("data_y_values", data_y[0,0], data_y[0,3], data_y[0,5], data_y[0,7])

    input_data = [data_x, data_y, data_w]

    return input_data
    

def get_data_xsec_grad_points(ip, morphing_weights_grad_points, data_w_eft):
    data_xsec_grad = np.array([np.dot(morphing_weights_grad_points[ip,:],data_w_eft[ie,:]) for ie in range(len(data_w_eft))]).sum(axis=0)
    return data_xsec_grad
    

#==================================================================================================
def evaluate_APSNN(input_data, model, i_eval, eval_step_size, criterion, parameters, stat_values, var_use, device, mode):

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
        data_loss_i = eval_data_w_sum*criterion(torch.IntTensor(eval_data_y), eval_data_yhat, torch.FloatTensor(eval_data_w).view(-1,1)).item()
        if parameters[4] == 'cce':
            data_acc_i = eval_data_w_sum*np.average(eval_data_y == eval_data_yhat.max(1)[1].numpy(), weights=eval_data_w)
        elif parameters[4] == 'bce':
            data_acc_i = eval_data_w_sum*np.average(eval_data_y == (eval_data_yhat[:, 0] > 0.5).numpy(), weights=eval_data_w)
        elif parameters[4] == 'mse':
            data_acc_i = 0
        del eval_data_yhat

        return [data_loss_i, data_acc_i]


#==================================================================================================
def feature_score_APSNN(input_data, model, min_loss, eval_step_size, criterion, parameters, variables, var_names, var_use, stat_values, device):

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
                i_eval_output = evaluate_APSNN(shuffled_data, model, i_eval, eval_step_size, criterion, parameters, stat_values, var_use, device, mode="metric")
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
def save_APSNN(model, model_outpath, dim, device):

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




















