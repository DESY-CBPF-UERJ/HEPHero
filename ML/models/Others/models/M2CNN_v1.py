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
import tools
import concurrent.futures
import multiprocessing
from itertools import repeat
from .ResNet import *
numpy_random = np.random.RandomState(16)

#==================================================================================================
def build_M2CNN(vec_variables, n_classes, parameters, stat_values, device):
    n_obj_types = len(vec_variables)
    n_channels = 3*n_obj_types

    resnet_arch = parameters[7][0]      
    if resnet_arch == 50:
        layers_arch = [3,4,6,3]
    elif resnet_arch == 101:
        layers_arch = [3,4,23,3]
    elif resnet_arch == 152:
        layers_arch = [3,8,36,3]
        
    model = ResNet(Bottleneck, layers_arch, n_classes, n_channels, parameters, stat_values, device)
    
    return model


#==================================================================================================
def model_parameters_M2CNN(param_dict):
    resnet_arch = param_dict["resnet_arch"]
    n_pixels_eta = param_dict["n_pixels_eta"]
    n_pixels_phi = param_dict["n_pixels_phi"]
    eta_cut = param_dict["eta_cut"]
    phi_cut = param_dict["phi_cut"]

    model_parameters = []
    for i_resnet_arch in resnet_arch:
        for i_n_pixels_eta in n_pixels_eta:
            for i_n_pixels_phi in n_pixels_phi:
                for i_eta_cut in eta_cut:
                    for i_phi_cut in phi_cut:
                        model_parameters.append([i_resnet_arch] + [i_n_pixels_eta] + [i_n_pixels_phi] + [i_eta_cut] + [i_phi_cut])

    return model_parameters


#==================================================================================================
def features_stat_M2CNN(train_data, test_data, vec_train_data, vec_test_data, vec_variables, vec_var_names, class_names, class_labels, class_colors, plots_outpath, parameters, load_it=None):

    n_obj_types = len(vec_variables)
    n_channels = 4*n_obj_types
    padding = 3
    n_pixels_eta = parameters[7][1]
    n_pixels_phi = parameters[7][2]
    eta_cut = parameters[7][3]
    phi_cut = parameters[7][4]
    
    position_train_data = {}
    tlorentz_train_data = {}
    position_test_data = {}
    tlorentz_test_data = {}
    n_nonzero_entries = 0
    sum_entry_values = 0
    for vecvar in vec_variables:
        dim1, dim2 = vec_train_data[vecvar].shape
        n_obj = int(dim2/6)
        vec_train_data[vecvar] = vec_train_data[vecvar].reshape(-1, n_obj, 6)
        vec_test_data[vecvar] = vec_test_data[vecvar].reshape(-1, n_obj, 6)

        position_train_data[vecvar] = vec_train_data[vecvar][:,:,0:2]
        tlorentz_train_data[vecvar] = vec_train_data[vecvar][:,:,2:6]
        del vec_train_data[vecvar]
        position_test_data[vecvar] = vec_test_data[vecvar][:,:,0:2]
        tlorentz_test_data[vecvar] = vec_test_data[vecvar][:,:,2:6]
        del vec_test_data[vecvar]

        tlorentz_train_data[vecvar][tlorentz_train_data[vecvar] == 0.] = 1
        tlorentz_test_data[vecvar][tlorentz_test_data[vecvar] == 0.] = 1

        tlorentz_train_data[vecvar] = np.log10(np.absolute(tlorentz_train_data[vecvar]))
        tlorentz_test_data[vecvar] = np.log10(np.absolute(tlorentz_test_data[vecvar]))

        n_nonzero_entries += np.count_nonzero(tlorentz_train_data[vecvar]) + np.count_nonzero(tlorentz_test_data[vecvar])
        sum_entry_values += tlorentz_train_data[vecvar].sum() + tlorentz_test_data[vecvar].sum()

        
        position_train_data[vecvar][:,:,0] = (position_train_data[vecvar][:,:,0]+phi_cut)*(1/((2*phi_cut)/n_pixels_phi))
        position_test_data[vecvar][:,:,0] = (position_test_data[vecvar][:,:,0]+phi_cut)*(1/((2*phi_cut)/n_pixels_phi))

        position_train_data[vecvar][:,:,1] = (position_train_data[vecvar][:,:,1]+eta_cut)*(1/((2*eta_cut)/n_pixels_eta))
        position_test_data[vecvar][:,:,1] = (position_test_data[vecvar][:,:,1]+eta_cut)*(1/((2*eta_cut)/n_pixels_eta))

        position_train_data[vecvar] = position_train_data[vecvar].astype(np.uint8)
        position_test_data[vecvar] = position_test_data[vecvar].astype(np.uint8)

        tlorentz_train_data[vecvar] = tlorentz_train_data[vecvar].astype(np.float16)
        tlorentz_test_data[vecvar] = tlorentz_test_data[vecvar].astype(np.float16)
    

    
    
    #print("position_train_data", position_train_data["lightJet_vector"][0:3,:,:])
    #print("tlorentz_train_data", tlorentz_train_data["lightJet_vector"][0:3,:,:])

    log_mean = sum_entry_values/n_nonzero_entries
    dim = (n_channels, n_pixels_eta+2*padding, n_pixels_phi+2*padding)
    
    np.set_printoptions(legacy='1.25')
    if load_it is None:
        print("log_mean: " + str(log_mean))
        print("dim: " + str(dim))
    stat_values={"log_mean": log_mean, "dim": dim}

    #for vecvar in vec_variables:
    #    tlorentz_train_data[vecvar] = tlorentz_train_data[vecvar]*(0.5/log_mean)

    #----------------------------------------------------------------------------------------------

    """
    cpu_count = multiprocessing.cpu_count()
    if cpu_count <= 2:
        cpu_count = 1
    else:
        cpu_count -= 2


    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        image_train_data_list = list(executor.map(get_image_data, range(n_obj_types), repeat(vec_variables), repeat(tlorentz_train_data), repeat(position_train_data), repeat(phi_cut), repeat(n_pixels_phi), repeat(eta_cut), repeat(n_pixels_eta)))
    image_train_data = np.concatenate(image_train_data_list, axis=0)

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        image_test_data_list = list(executor.map(get_image_data, range(n_obj_types), repeat(vec_variables), repeat(tlorentz_test_data), repeat(position_test_data), repeat(phi_cut), repeat(n_pixels_phi), repeat(eta_cut), repeat(n_pixels_eta)))
    image_test_data = np.concatenate(image_test_data_list, axis=0)
    """

    image_train_data = []
    for vecvar in vec_variables:
        for itl in range(4):
            dim1, dim2, dim3 = tlorentz_train_data[vecvar].shape
            image_train_data.append(np.zeros((dim1,n_pixels_eta,n_pixels_phi)))
            for ievt in range(dim1): # paralell processing
                for iobj in range(dim2):
                    if tlorentz_train_data[vecvar][ievt,iobj,:].sum() != 0:
                        value = tlorentz_train_data[vecvar][ievt,iobj,itl]
                        iphi = position_train_data[vecvar][ievt,iobj,0]
                        #step_phi = (2*phi_cut)/n_pixels_phi
                        #iphi = int((phi+phi_cut)/step_phi)
                        ieta = position_train_data[vecvar][ievt,iobj,1]
                        #step_eta = (2*eta_cut)/n_pixels_eta
                        #ieta = int((eta+eta_cut)/step_eta)
                        image_train_data[-1][ievt,ieta,iphi] = value

    image_test_data = []
    for vecvar in vec_variables:
        for itl in range(4):
            dim1, dim2, dim3 = tlorentz_test_data[vecvar].shape
            image_test_data.append(np.zeros((dim1,n_pixels_eta,n_pixels_phi)))
            for ievt in range(dim1): # paralell processing
                for iobj in range(dim2):
                    if tlorentz_test_data[vecvar][ievt,iobj,:].sum() != 0:
                        value = tlorentz_test_data[vecvar][ievt,iobj,itl]
                        iphi = position_test_data[vecvar][ievt,iobj,0]
                        #step_phi = (2*phi_cut)/n_pixels_phi
                        #iphi = int((phi+phi_cut)/step_phi)
                        ieta = position_test_data[vecvar][ievt,iobj,1]
                        #step_eta = (2*eta_cut)/n_pixels_eta
                        #ieta = int((eta+eta_cut)/step_eta)
                        image_test_data[-1][ievt,ieta,iphi] = value
    

    image_train_data = np.array(image_train_data, dtype=np.float16)
    image_test_data = np.array(image_test_data, dtype=np.float16)

    image_train_data = np.transpose(image_train_data, axes=[1, 0, 2, 3])
    image_test_data = np.transpose(image_test_data, axes=[1, 0, 2, 3])
    
    print("image_train_data", image_train_data.shape)
    print("image_test_data", image_test_data.shape)
    #print("image_train_data", image_train_data[0:1,:,:,:])

    for vecvar in vec_variables:
        itl_name = ["logE", "logPx", "logPy", "logPz"]
        for itl in range(4):
            
            fig1 = plt.figure(figsize=(9,5))
            gs1 = gs.GridSpec(1, 1)
            #==================================================
            ax1 = plt.subplot(gs1[0])
            #==================================================
            tlvar = itl_name[itl]
            bins = np.linspace(0,log_mean*2.5,101)
            for ikey in range(len(class_names)):

                train_var_array = tlorentz_train_data[vecvar][train_data["class"] == ikey]
                train_mvaW_array = train_data["mvaWeight"][train_data["class"] == ikey]
                train_mvaW_array = np.transpose(np.transpose(np.ones_like(train_var_array), axes=[1, 2, 0])*np.array(train_mvaW_array), axes=[2, 0, 1])          
                train_mvaW_array = train_mvaW_array[:,:,itl][train_var_array[:,:,0] > 0]
                train_var_array = train_var_array[:,:,itl][train_var_array[:,:,0] > 0]
                df_train_var = {tlvar: train_var_array, 'mvaWeight': train_mvaW_array}
                tools.step_plot( ax1, tlvar, df_train_var, label=class_labels[ikey]+" (train)", color=class_colors[ikey], weight="mvaWeight", bins=bins, error=True )
                
                test_var_array = tlorentz_test_data[vecvar][test_data["class"] == ikey]
                test_mvaW_array = test_data["mvaWeight"][test_data["class"] == ikey]
                test_mvaW_array = np.transpose(np.transpose(np.ones_like(test_var_array), axes=[1, 2, 0])*np.array(test_mvaW_array), axes=[2, 0, 1])          
                test_mvaW_array = test_mvaW_array[:,:,itl][test_var_array[:,:,0] > 0]
                test_var_array = test_var_array[:,:,itl][test_var_array[:,:,0] > 0]
                df_test_var = {tlvar: test_var_array, 'mvaWeight': test_mvaW_array}
                tools.step_plot( ax1, tlvar, df_test_var, label=class_labels[ikey]+" (test)", color=class_colors[ikey], weight="mvaWeight", bins=bins, error=True, linestyle='dotted' )
                
                
            ax1.set_xlabel(tlvar, size=14, horizontalalignment='right', x=1.0)
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
                plt.savefig(os.path.join(plots_outpath, vecvar +"_"+ tlvar + '.png'), dpi=400)
                plt.savefig(os.path.join(plots_outpath, vecvar +"_"+ tlvar + '.pdf'))
            else:
                plt.savefig(os.path.join(plots_outpath, vecvar +"_"+ tlvar +"_"+ str(load_it) + '.png'), dpi=400)
                plt.savefig(os.path.join(plots_outpath, vecvar +"_"+ tlvar +"_"+ str(load_it) + '.pdf'))
            plt.close()

    return stat_values
    
    
#==================================================================================================
def update_M2CNN(model, criterion, parameters, batch_data, device):
    
    if parameters[3] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters[6], eps=1e-07)
        # lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
    elif parameters[3] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=parameters[6])
        # lr=?, momentum=0, dampening=0, weight_decay=0, nesterov=False
    elif parameters[3] == "ranger":
        optimizer = Ranger(model.parameters(), lr=parameters[6])
    #print(optimizer.state_dict())

    image_data_ext, data_y_b, data_w_b = batch_data
    del batch_data

    if device == "cuda":
        w = torch.FloatTensor(data_w_b).view(-1,1).to("cuda")
        y = torch.IntTensor(data_y_b).view(-1,1).to("cuda")
        image_data_ext = torch.FloatTensor(image_data_ext).to("cuda")
    else:
        w = torch.FloatTensor(data_w_b).view(-1,1)
        y = torch.IntTensor(data_y_b).view(-1,1)
        image_data_ext = torch.FloatTensor(image_data_ext)

    image_data_ext.requires_grad=True
    yhat = model(image_data_ext)

    loss = criterion(y, yhat, w, device=device)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return model


#==================================================================================================
def process_data_M2CNN(scalar_var, vector_var, vec_variables, parameters):

    n_obj_types = len(vec_variables)
    n_channels = 4*n_obj_types
    n_pixels_eta = parameters[7][1]
    n_pixels_phi = parameters[7][2]
    eta_cut = parameters[7][3]
    phi_cut = parameters[7][4]
    
    scalar_var = pd.DataFrame.from_dict(scalar_var)
    data_y = np.array(scalar_var['class']).ravel()
    data_w = np.array(scalar_var['mvaWeight']).ravel()

    position_data = {}
    tlorentz_data = {}
    for vecvar in vec_variables:
        dim1, dim2 = vector_var[vecvar].shape
        n_obj = int(dim2/6)
        vector_var[vecvar] = vector_var[vecvar].reshape(-1, n_obj, 6)

        position_data[vecvar] = vector_var[vecvar][:,:,0:2]
        tlorentz_data[vecvar] = vector_var[vecvar][:,:,2:6]
        del vector_var[vecvar]

        tlorentz_data[vecvar][tlorentz_data[vecvar] == 0.] = 1
        tlorentz_data[vecvar] = np.log10(np.absolute(tlorentz_data[vecvar]))

        position_data[vecvar][:,:,0] = (position_data[vecvar][:,:,0]+phi_cut)*(1/((2*phi_cut)/n_pixels_phi))
        position_data[vecvar][:,:,1] = (position_data[vecvar][:,:,1]+eta_cut)*(1/((2*eta_cut)/n_pixels_eta))
        
        position_data[vecvar] = position_data[vecvar].astype(np.uint8)
        tlorentz_data[vecvar] = tlorentz_data[vecvar].astype(np.float16)
        
    #----------------------------------------------------------------------------------------------
    

    """
    cpu_count = multiprocessing.cpu_count()
    if cpu_count <= 2:
        cpu_count = 1
    else:
        cpu_count -= 2

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        image_data_list = list(executor.map(get_image_data, range(n_obj_types), repeat(vec_variables), repeat(tlorentz_data), repeat(position_data), repeat(phi_cut), repeat(n_pixels_phi), repeat(eta_cut), repeat(n_pixels_eta)))
    image_data = np.concatenate(image_data_list, axis=0)
    """
    
    image_data = []
    for vecvar in vec_variables:
        for itl in range(4):        
            dim1, dim2, dim3 = tlorentz_data[vecvar].shape
            image_data.append(np.zeros((dim1,n_pixels_eta,n_pixels_phi)))
            for iobj in range(dim2):
                for ievt in range(dim1): 
                    if tlorentz_data[vecvar][ievt,iobj,:].sum() != 0:
                        value = tlorentz_data[vecvar][ievt,iobj,itl]
                        iphi = position_data[vecvar][ievt,iobj,0]
                        #step_phi = (2*phi_cut)/n_pixels_phi
                        #iphi = int((phi+phi_cut)/step_phi)
                        ieta = position_data[vecvar][ievt,iobj,1]
                        #step_eta = (2*eta_cut)/n_pixels_eta
                        #ieta = int((eta+eta_cut)/step_eta)
                        image_data[-1][ievt,ieta,iphi] = value                        
        del tlorentz_data[vecvar]
        del position_data[vecvar]

    image_data = np.array(image_data, dtype=np.float16)
    image_data = np.transpose(image_data, axes=[1, 0, 2, 3])
    print("image_data", image_data.dtype)
    #----------------------------------------------------------------------------------------------
    padding = 3
    
    colF_data = image_data[:, :, :, 0:padding]
    colF_data = colF_data.reshape(-1, n_channels, n_pixels_eta, padding)
    #print("colF_data", colF_data.shape)
    image_data_ext = np.concatenate((image_data, colF_data), axis=3)

    colL_data = image_data[:, :, :, -padding:]
    colL_data = colL_data.reshape(-1, n_channels, n_pixels_eta, padding)
    #print("colL_data", colL_data.shape)
    image_data_ext = np.concatenate((colL_data, image_data_ext), axis=3)
    
    row0_data = np.zeros_like(image_data_ext[:, :, 0:padding, :])
    row0_data = row0_data.reshape(-1, n_channels, padding, n_pixels_phi+(2*padding))
    #print("row0_data", row0_data.shape)
    image_data_ext = np.concatenate((row0_data, image_data_ext, row0_data), axis=2)

    #print("image_data_ext", image_data_ext.shape)
    #print("data_y", data_y.shape)
    #print("data_w", data_w.shape)
    #----------------------------------------------------------------------------------------------

    input_data = [image_data_ext, data_y, data_w]

    return input_data


#==================================================================================================
def evaluate_M2CNN(input_data, model, i_eval, eval_step_size, criterion, parameters, device, mode):
    
    image_data_ext, data_y, data_w = input_data

    n_eval_steps = int(len(data_w)/eval_step_size) + 1
    last_eval_step = len(data_w)%eval_step_size

    if i_eval < n_eval_steps-1:
        eval_image_data_ext = image_data_ext[i_eval*eval_step_size:(i_eval+1)*eval_step_size,:,:,:]
        eval_data_y = data_y[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
        eval_data_w = data_w[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
    elif last_eval_step > 0:
        eval_image_data_ext = image_data_ext[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_step,:,:,:]
        eval_data_y = data_y[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_step]
        eval_data_w = data_w[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_step]
    else:
        return None

    if device == "cuda":
        eval_data_yhat = model(torch.FloatTensor(eval_image_data_ext).to("cuda"))
        eval_data_yhat = eval_data_yhat.cpu()
    else:
        eval_data_yhat = model(torch.FloatTensor(eval_image_data_ext))

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
def feature_score_M2CNN(input_data, model, min_loss, eval_step_size, criterion, parameters, vec_variables, vec_var_names, device):

    n_eval_steps = int(len(input_data[-1])/eval_step_size) + 1
    data_w_sum = input_data[-1].sum()

    tl_name = ["logE", "logPx", "logPy", "logPz"]
    np.set_printoptions(threshold=sys.maxsize)
    
    features_score = []
    features_score_unc = []
    features_names = []
    for ivar in tqdm(range(len(vec_variables))):
        for itl in range(4):

            
            losses = []
            for irep in range(30):
    
                data_x, data_y, data_w = input_data
                data_x_shuffled = data_x.copy()
                #if irep == 0:
                #    print("data_x_shuffled.shape", data_x_shuffled.shape)
                #    print("data_x_shuffled[0,4*ivar+itl,:,:]", data_x_shuffled[0,4*ivar+itl,:,:])
                #    print("data_x_shuffled[0,4*ivar+(itl+1),:,:]", data_x_shuffled[0,4*ivar+(itl+1),:,:])
                numpy_random.shuffle(data_x_shuffled[:,4*ivar+itl,:,:])
                #if irep == 0:
                #    print("data_x_shuffled.shape", data_x_shuffled.shape)
                #    print("data_x_shuffled[0,4*ivar+itl,:,:]", data_x_shuffled[0,4*ivar+itl,:,:])
                #    print("data_x_shuffled[0,4*ivar+(itl+1),:,:]", data_x_shuffled[0,4*ivar+(itl+1),:,:])
                #    print("")
                shuffled_data = [data_x_shuffled, data_y, data_w]
    
                data_loss_i = 0
                for i_eval in range(n_eval_steps):
                    i_eval_output = evaluate_M2CNN(shuffled_data, model, i_eval, eval_step_size, criterion, parameters, device, mode="metric")
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
            features_names.append(vec_var_names[ivar]+"_"+tl_name[itl])

    feature_score_info = features_score, features_score_unc, features_names

    return feature_score_info


#==================================================================================================
def save_M2CNN(model, model_outpath, dim, device):
    
    #ONNX
    input_names = ['features']
    output_names = ['output']
    dynamic_axes = {'features': {0: 'N'}, 'output': {0: 'N'}}
    input_shapes = {'features': (1, dim[0], dim[1], dim[2])}
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

    
#==================================================================================================
def get_image_data(ivecvar, vec_variables, tlorentz_data, position_data, phi_cut, n_pixels_phi, eta_cut, n_pixels_eta):
    image_ivecvar_data = []
    for itl in range(4):
        dim1, dim2, dim3 = tlorentz_data[vec_variables[ivecvar]].shape
        image_ivecvar_data.append(np.zeros((dim1,n_pixels_eta,n_pixels_phi)))
        for ievt in range(dim1): # paralell processing
            for iobj in range(dim2):
                if tlorentz_data[vec_variables[ivecvar]][ievt,iobj,:].sum() != 0:
                    value = tlorentz_data[vec_variables[ivecvar]][ievt,iobj,itl]
                    phi = position_data[vec_variables[ivecvar]][ievt,iobj,0]
                    step_phi = (2*phi_cut)/n_pixels_phi
                    iphi = int((phi+phi_cut)/step_phi)
                    eta = position_data[vec_variables[ivecvar]][ievt,iobj,1]
                    step_eta = (2*eta_cut)/n_pixels_eta
                    ieta = int((eta+eta_cut)/step_eta)
                    image_ivecvar_data[-1][ievt,ieta,iphi] = value

    return image_ivecvar_data