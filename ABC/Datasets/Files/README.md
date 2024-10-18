# Datasets

**Instructions to produce text files with dataset paths**

* `The production of the text files is done in the lxplus.`
* `The list of datasets used in the analysis are defined inside the python scripts below.`

# Get permission to access datasets

Type the command below and enter your GRID password:
```bash
voms-proxy-init --voms cms
```

# Check datasets

# Check datasets

Check if datsets of the analysis are available in a specific **year** and **version**:
```bash
python check_data.py -y period
python check_mc.py -y period -v version
```

For simulation, the NANOAOD **version** can be specified as well:
```bash
python check_mc.py -y period -v version
```

years = 16, 17, 18, 22, 23
versions = 9, 12

# Get dataset paths

The text files are produced running the scripts below :   
```bash
python get_signal.py -p period -v version
python get_bkg.py -p period -v version
python get_data_16.py -v version
python get_data_17.py -v version
python get_data_18.py -v version
```
A **period** consists of one **year** plus a **dti** (data taking interval) in the format **dti_year**:
periods = 0_16, 1_16, 0_17, 0_18, 0_22, 1_22, 0_23, 1_23


For convenience, edit and run the **get_files.sh** script to produce text files for all datasets of interest.
```bash
./get_files.sh 
```
The production is a time consuming process. 





