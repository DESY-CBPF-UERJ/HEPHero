#!/bin/bash
import os
import sys
import argparse


#======GET SETUP FILE==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--analysis", type=str, default="GEN")
args = parser.parse_args()
print('runSelection.py set to ' + args.analysis)
print('')
   
sys.path.insert(0, 'setups')
sm = __import__(args.analysis)
   

#======CREATE NEW SETUP============================================================================
with open("runSelection_temp.py", "w") as newfile:
    newfile.write("#!/bin/bash\n")
    newfile.write("import os\n")
    newfile.write("import time\n")
    newfile.write("import sys\n")
    newfile.write("import argparse\n")
    newfile.write("import warnings\n")
    newfile.write("import json\n")
    newfile.write("from shutil import copyfile\n")
    newfile.write("import glob\n")    
    newfile.write("import glob\n")
    newfile.write("import glob\n")
    newfile.write("import ast\n")
    newfile.write("import h5py\n")
    newfile.write("import pandas as pd\n")
    newfile.write("import numpy as np\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Main Setup\n")    
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("selection = '" + sm.selection +"'\n")
    newfile.write("analysis = '" + sm.analysis +"'\n")
    newfile.write("treeName = '" + sm.treeName +"'\n")
    newfile.write("LumiWeights = " + str(sm.LumiWeights) +"\n")
    newfile.write("\n")
    newfile.write("analysis_parameters = {\n")
    for param in sm.analysis_parameters:
        newfile.write("'" + param + "': " + str(sm.analysis_parameters[param]) +",\n")
    newfile.write("}\n")
    newfile.write("\n")
    newfile.write("corrections = {  # 0-don't apply, 1-apply\n")
    for corr in sm.corrections:
        newfile.write("'" + corr + "': " + str(sm.corrections[corr]) +",\n")
    newfile.write("}\n")
    newfile.write("\n")    
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Systematics\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("lateral_systematics = {\n")
    for source in sm.lateral_systematics:
        newfile.write("'" + source + "': " + str(sm.lateral_systematics[source]) +",\n")
    newfile.write("}\n")    
    newfile.write("\n")
    newfile.write("vertical_systematics = {\n")
    for source in sm.vertical_systematics:
        newfile.write("'" + source + "': " + str(sm.vertical_systematics[source]) +",\n")
    newfile.write("}\n")
    newfile.write("\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Jobs setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("NumMaxEvents = " + str(sm.NumMaxEvents) +"\n")
    if args.analysis != "GEN":
        newfile.write("NumFilesPerJob_Data = " + str(sm.NumFilesPerJob_Data) +"\n")
        newfile.write("NumFilesPerJob_Signal = " + str(sm.NumFilesPerJob_Signal) +"\n")
        newfile.write("NumFilesPerJob_Bkg = " + str(sm.NumFilesPerJob_Bkg) +"\n")
    newfile.write("\n")    
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Datasets\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    if args.analysis == "GEN":
        newfile.write("datasets = [\n")
        for dataset in sm.datasets:
            newfile.write(str(dataset) + ",\n")
        newfile.write("]\n")
    elif args.analysis == "OPENDATA":
        newfile.write("sys.path.insert(0, 'Datasets')\n")
        newfile.write("from Signal import *\n")
        newfile.write("from Bkg import *\n")
        newfile.write("from Data import *\n")
        newfile.write("datasets = []\n")
        newfile.write("\n")
        for dataset in sm.datasets:
            newfile.write("datasets.extend(" + dataset + "_12" + ")\n")
    else:
        newfile.write("sys.path.insert(0, 'Datasets')\n")
        newfile.write("from Signal import *\n")
        newfile.write("from Bkg import *\n")
        newfile.write("from Data import *\n")
        newfile.write("datasets = []\n") 
        periods = ["preVFP_16", "postVFP_16", "17", "18"]
        for period in periods:
            newfile.write("\n")
            for dataset in sm.datasets:
                newfile.write("datasets.extend(" + dataset + "_" + period + ")\n")
    newfile.write("\n")    
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Metadata\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("metadata = {\n")
    for data in sm.metadata:
        newfile.write("'" + data + "': '" + str(sm.metadata[data]) +"',\n")
    newfile.write("}\n")
    newfile.write("\n")    
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Plots\n")    
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("Get_Image_in_EPS = " + str(sm.Get_Image_in_EPS) +"\n")
    newfile.write("Get_Image_in_PNG = " + str(sm.Get_Image_in_PNG) +"\n")
    newfile.write("Get_Image_in_PDF = " + str(sm.Get_Image_in_PDF) +"\n")
    newfile.write("\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")


#======ADD FIXED PART==============================================================================
linelist = open("runSelection.py").readlines()

with open("runSelection_temp.py", "a") as newfile:
    flag = 0
    for line in linelist:
        if line.startswith("# [DO NOT TOUCH THIS PART]"):
            flag = 1
        if flag:
            newfile.write(line)
        

#======REPLACE OLD RUNSELECTION BY THE NEW NEW=====================================================
os.system("mv runSelection_temp.py runSelection.py")


#======CREATE NEW ANALYSIS FILE IN TOOLS===========================================================
with open("tools/analysis.txt", "w") as txtfile:
    txtfile.write(sm.analysis)
