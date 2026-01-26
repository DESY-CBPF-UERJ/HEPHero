#!/bin/bash
import os
import sys
import argparse
from shutil import copyfile


#======GET SETUP FILE==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="NN")
parser.add_argument("-t", "--tag", type=str, default="template")
args = parser.parse_args()

cwd_path = os.getcwd()
cwd_last_folder = os.getcwd().split("/")[-1]

if cwd_last_folder == "ML":
    sys.path.insert(0, 'models/'+args.model)
    with open('../tools/analysis.txt') as f:
        analysis = f.readline()
    analysis_path = "../"+analysis+"/ML"
    submit_file = "submit_jobs.sh"
    condor_file = "condor.py"
elif cwd_last_folder == "HEPHero":
    sys.path.insert(0, 'ML/models/'+args.model)
    with open('tools/analysis.txt') as f:
        analysis = f.readline()
    analysis_path = analysis+"/ML"
    submit_file = "ML/submit_jobs.sh"
    condor_file = "ML/condor.py"
else:
    sys.exit("You must run generate_tariner.py from the HEPHero or HEPHero/ML directories.")

sm = __import__('setup')

if not os.path.exists(analysis_path):    
    os.makedirs(analysis_path)
    copyfile(submit_file, analysis_path+"/submit_jobs.sh")
    copyfile(condor_file, analysis_path+"/condor.py")

trainer_name = 'train_' + args.model + '_' + args.tag + '.py'
trainer_file = os.path.join(analysis_path, trainer_name)

if os.path.exists(trainer_file):
    sys.exit("There is already a trainer file named " + trainer_file + ". Try a different name or delete the current file.")

print('Generate ' + trainer_name + " at " + analysis_path)
print('')

#======CREATE NEW SETUP============================================================================
with open(trainer_file, "w") as newfile:
    newfile.write("import sys\n")
    newfile.write("import numpy as np\n")
    newfile.write("import pandas as pd\n")
    newfile.write("import os\n")
    newfile.write("import time\n")
    newfile.write("from tqdm import tqdm\n")
    newfile.write("import concurrent.futures as cf\n")
    newfile.write("import argparse\n")
    newfile.write("import matplotlib.pyplot as plt\n")
    newfile.write("import matplotlib.gridspec as gs\n")
    newfile.write("from matplotlib.ticker import AutoMinorLocator\n")
    newfile.write("import json\n")
    newfile.write("sys.path.insert(0, '../../ML')\n")
    newfile.write("import tools\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# General Setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("input_path = '" + sm.input_path +"'\n")
    newfile.write("output_path = '" + sm.output_path +"'\n")
    newfile.write("periods = " + str(sm.periods) +"\n")
    newfile.write("tag = '" + args.tag +"'\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# ML setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("device = '" + sm.device +"'\n")
    newfile.write("library = '" + sm.library +"'\n")
    newfile.write("optimizer = " + str(sm.optimizer) +"\n")
    newfile.write("loss_func = " + str(sm.loss_func) +"\n")
    newfile.write("learning_rate = " + str(sm.learning_rate) +"\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Models setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("model_type = '" + sm.model_type +"'\n")
    newfile.write("model_parameters = " + str(sm.model_parameters) +"\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Training setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("batch_size = " + str(sm.batch_size) +"\n")
    newfile.write("load_size_stat = " + str(sm.load_size_stat) +"\n")
    newfile.write("load_size_training = " + str(sm.load_size_training) +"\n")
    newfile.write("num_load_for_check = " + str(sm.num_load_for_check) +"\n")
    newfile.write("train_frac = " + str(sm.train_frac) +"\n")
    newfile.write("eval_step_size = " + str(sm.eval_step_size) +"\n")
    newfile.write("eval_interval = " + str(sm.eval_interval) +"\n")
    newfile.write("num_max_iterations = " + str(sm.num_max_iterations) +"\n")
    newfile.write("early_stopping = " + str(sm.early_stopping) +"\n")
    newfile.write("initial_model_path = " + str(sm.initial_model_path) +"\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Inputs setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("feature_info = " + str(sm.feature_info) +"\n")
    newfile.write("\n")
    newfile.write("scalar_variables = " + str(sm.scalar_variables) +"\n")
    newfile.write("\n")
    newfile.write("vector_variables = " + str(sm.vector_variables) +"\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Preprocessing setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("reweight_variables = " + str(sm.reweight_variables) +"\n")
    newfile.write("normalization_method = '" + sm.normalization_method +"'\n")
    newfile.write("\n")
    newfile.write("pca_transformation = " + str(sm.pca_transformation) +"\n")
    newfile.write("pca_custom_classes = " + str(sm.pca_custom_classes) +"\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Classes setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("classes = " + str(sm.classes) +"\n")
    newfile.write("\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")


#======ADD FIXED PART==============================================================================
if cwd_last_folder == "ML":
    linelist = open("config_train.py").readlines()
elif cwd_last_folder == "HEPHero":
    linelist = open("ML/config_train.py").readlines()

with open(trainer_file, "a") as newfile:
    flag = 0
    for line in linelist:
        if line.startswith("# [DO NOT TOUCH THIS PART]"):
            flag = 1
        if flag:
            newfile.write(line)
        
