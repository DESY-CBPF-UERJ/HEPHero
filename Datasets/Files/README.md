# Datasets

**Instructions to produce text files with dataset paths**

* `The production of the text files is done in the lxplus. However, the setup is valid to DESY as well.` 
* `The list of datasets used in the analysis are defined inside the python scripts below.`

# Get permission to access datasets

Type the command below and enter your GRID password:
```bash
voms-proxy-init --voms cms
```

# Check datasets

Check if datsets of the analysis are available in a specific **period** and **version**:   
```bash
python check_data.py -p period -v version
python check_mc.py -p period -v version
```
period = 16, 17, 18   
version = 8, 9   

# Get dataset paths

The text files are produced running the scripts below :   
```bash
python get_signal.py -p period -v version
python get_bkg.py -p period -v version
python get_data_16.py -v version
python get_data_17.py -v version
python get_data_18.py -v version
```
Use the flag **apv** to produce text files for 2016_preVFP as in the example below:  
```bash
python get_bkg.py -p 16 -v version --apv 
```

For convenience, edit and run the **get_files.sh** script to produce text files for all datasets of interest.
```bash
./get_files.sh 
```
The production is a time consuming process. 





