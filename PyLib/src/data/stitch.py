import os

import pandas as pd
import numpy as np

"""   
def stitch_datasets( ds, md, process_list, prob_list, var, weight, limits, alpha=False, process_list_ext=[] ):
    
    ds_list = []
    N_list = []
    for i in range(len(process_list)):
        ds_list.append(ds[process_list[i]])
        N_list.append(md[process_list[i]]["SumGenWeights"])
        
    weight_list = []
    weight_list.append(1)
    
    for i in range(len(process_list)):
        if i > 0:
            weight_list.append((N_list[0]*prob_list[i])/(N_list[0]*prob_list[i] + N_list[i]*(md[process_list[0]]["CrossSection"]/md[process_list[i]]["CrossSection"])))

    for i in range(len(process_list)):
        if i > 0:
            ds_list[0].loc[(ds_list[0][var] >= limits[i]) & (ds_list[0][var] < limits[i+1]), weight] =  ds_list[0][(ds_list[0][var] >= limits[i]) & (ds_list[0][var] < limits[i+1])][weight]*weight_list[i]
            ds_list[i].loc[:, weight] = ds_list[i][weight]*weight_list[i]

    return weight_list
"""        


def stitch_datasets( ds, md, process_list, prob_list, var, weight, limits, alpha=False, process_list_ext=[] ):
    
    ds_list = []
    N_list = []
    for i in range(len(process_list)):
        ds_list.append(ds[process_list[i]])
        if len(process_list_ext) == 0:
            N_list.append(md[process_list[i]]["SumGenWeights"]*md[process_list[i]]["lumiWeight"])
        elif len(process_list_ext) <= len(process_list):
            print("Error: process_list_ext size must be bigger that process_list size")
        else:
            if i < len(process_list)-1:
                N_list.append(md[process_list[i]]["SumGenWeights"]*md[process_list[i]]["lumiWeight"])
            else:
                n_ext = len(process_list_ext) - len(process_list)
                N_i = 0
                for j in range(n_ext):
                    N_i = N_i + md[process_list_ext[i+j]]["SumGenWeights"]*md[process_list_ext[i+j]]["lumiWeight"]
                N_list.append(N_i)

    weight_list = []
    weight_list.append(1)
    
    if alpha:
        alpha_list = []
        alpha_list.append(1)
        for i in range(len(process_list)):
            if i > 0:
                #alpha_i = len(ds_list[0][(ds_list[0][var] >= limits[i]) & (ds_list[0][var] < limits[i+1])])/len(ds_list[i])
                alpha_i = (prob_list[i]*md[process_list[0]]["SumGenWeights"]/md[process_list[0]]["CrossSection"])/(md[process_list[i]]["SumGenWeights"]/md[process_list[i]]["CrossSection"])
                alpha_list.append(alpha_i)
                weight_list.append((N_list[0]*prob_list[i])/(N_list[0]*prob_list[i]*alpha_i + N_list[i]))

        for i in range(len(weight_list)):
            if i > 0:
                ds_list[0].loc[(ds_list[0][var] >= limits[i]) & (ds_list[0][var] < limits[i+1]), weight] =  ds_list[0][(ds_list[0][var] >= limits[i]) & (ds_list[0][var] < limits[i+1])][weight]*weight_list[i]*alpha_list[i]
                ds_list[i].loc[:, weight] = ds_list[i][weight]*weight_list[i]
        return weight_list, alpha_list
    else:
        for i in range(len(process_list)):
            if i > 0:
                weight_list.append((N_list[0]*prob_list[i])/(N_list[0]*prob_list[i] + N_list[i]))

        for i in range(len(weight_list)):
            if i > 0:
                ds_list[0].loc[(ds_list[0][var] >= limits[i]) & (ds_list[0][var] < limits[i+1]), weight] =  ds_list[0][(ds_list[0][var] >= limits[i]) & (ds_list[0][var] < limits[i+1])][weight]*weight_list[i]
                ds_list[i].loc[:, weight] = ds_list[i][weight]*weight_list[i]

        return weight_list
   
    
    
