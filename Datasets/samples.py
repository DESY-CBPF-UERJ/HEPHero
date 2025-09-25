import os
import sys

files_dir = "../AP_Template_R3/Datasets"
sys.path.insert(0, files_dir)
from Data import *

def get_samples( analysis, basedir, period, group_data=False ):

    sys.path.insert(0, '../'+analysis+'/Datasets')
    from Bkg import nano_version

    year = period[-2:]
    dti = period[0]
    text_basedir = os.getcwd()[:-6]
    list_basedir = os.listdir(basedir)
    periodTag = period + "_files"

    path_bkg = text_basedir+'/'+analysis+'/Datasets/Files/bkg_'+year+'/dti_'+dti+'/'+nano_version+'/'
    path_signal = text_basedir+'/'+analysis+'/Datasets/Files/signal_'+year+'/dti_'+dti+'/'+nano_version+'/'
    path_data = text_basedir+'/'+analysis+'/Datasets/Files/data_'+year+'/'+nano_version+'/'

    list_bkg = os.listdir(path_bkg)
    list_signal = os.listdir(path_signal)
    list_data = os.listdir(path_data)
    
    list_bkg = [name[:-4] for name in list_bkg]
    list_signal = [name[:-4] for name in list_signal]
    list_data = [name[:-4] for name in list_data]
    
    samples = {}
    for sample_name in list_bkg:
        samples[sample_name] = [i for i in list_basedir if sample_name == i.split("_"+periodTag)[0]]
    for sample_name in list_signal:
        samples[sample_name] = [i for i in list_basedir if sample_name == i.split("_"+periodTag)[0]]
    if group_data:
        for era in eras[period]:
            samples["Data_"+era] = [i for i in list_basedir if "Data" in i and "_"+era+"_" in i and periodTag in i]
    else:    
        for sample_name in list_data:
            samples[sample_name] = [i for i in list_basedir if sample_name == i.split("_"+periodTag)[0]]
    
    empty_samples = []
    for sample in samples:
        if len(samples[sample]) == 0:
            empty_samples.append(sample)
    for sample in empty_samples:
        del samples[sample]

    return samples


