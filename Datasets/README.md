# Datasets

**Instructions to produce text files with root file paths**

* `The production of the text files is done in the lxplus.`
* `The list of datasets are defined inside the python scripts at ../<Analysis Project>/Datasets/Files.`
* `The text files produced will also be stored at ../<Analysis Project>/Datasets/Files.`
* `To run the python scripts below, you can NOT make the hepenv setup.`

# Get permission to access datasets

Type the command below and enter your GRID password:  
```bash
voms-proxy-init --voms cms
```

# Check datasets

Check if MC datsets of the analysis are available for a campaign containing a specific **tag** in its name:
```bash
python3 check_bkg.py -t tag
python3 check_signal.py -t tag
```
For DATA datasets, the **YEAR** must be specified:
```bash
python3 check_data.py -y year -t tag
```

# Produce text files containing the root file paths

The text files are produced running the scripts below :  
```bash
python3 get_signal.py -p period -v version
python3 get_bkg.py -p period -v version
python3 get_data.py -y year -v version
```
A **period** consists of one **year** plus a **dti** (data taking interval) in the format **dti_year**:  
Example of periods = 0_16, 1_16, 0_17, 0_18, 0_22, 1_22, 0_23, 1_23




