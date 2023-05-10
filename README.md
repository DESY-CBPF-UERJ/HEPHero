# ![HEPHERO](Metadata/logoframe.svg)

**HEPHero - Framework for the DESY-CBPF-UERJ collaboration**

# Initial setup

Set up your environment: (only once in your life)   
CERN - add the lines below to the file **/afs/cern.ch/user/${USER:0:1}/${USER}/.bashrc** and restart your session.   
DESY - add the lines below to the file **/afs/desy.de/user/${USER:0:1}/${USER}/.zshrc** and restart your session. (Create the file if it doesn't exist)   
UERJ - add the lines below to the file **/home/${USER}/.bashrc** and restart your session.   

```bash
export HEP_OUTPATH=<place the full path to a directory that will store the outputs>
export REDIRECTOR=<place the redirector suitable to your geographic region>
export MACHINES=<place the organization name, owner of the computing resources>

alias hepenv='source /afs/cern.ch/work/g/gcorreia/public/hepenv_setup.sh'  #(CERN)
alias hepenv='source /afs/desy.de/user/g/gcorreia/public/hepenv_setup.sh'  #(DESY)
alias hepenv='source /mnt/hadoop/cms/store/user/gcorreia/hepenv_setup.sh'  #(UERJ)
```

Possible outpaths:   

* `At CERN, use a folder inside your eos area` 
* `At DESY, use a folder inside your dust area`
* `At UERJ, you must create and define your outpath as: /home/username/output`

Possible redirectors (only used at CERN):   

* `cmsxrootd.fnal.gov` (USA)
* `xrootd-cms.infn.it` (Europe/Asia)
* `cms-xrd-global.cern.ch` (Global)

Possible machines:   

* `CERN`
* `DESY`
* `UERJ`

# Examples

```bash
export HEP_OUTPATH=/eos/user/g/gcorea/output
export REDIRECTOR=xrootd-cms.infn.it
export MACHINES=CERN
```

```bash
export HEP_OUTPATH=/nfs/dust/cms/user/gcorea/output
export REDIRECTOR=None
export MACHINES=DESY
```

```bash
export HEP_OUTPATH=/home/gcorea/output  
export REDIRECTOR=None
export MACHINES=UERJ
```

# Quick start

Inside your private area (NOT in the eos or dust area and NOT inside a CMSSW release), download the code. 

```bash
git clone git@github.com:DESY-CBPF-UERJ/HEPHero.git
```

Source the hepenv environment before work with the HEPHero:
```
hepenv
```

Enter in the HEPHero directory and compile the code (running cmake is necessary only at the first time):

```bash
cd HEPHero
cmake .
make -j 8
```

Set up the runSelection.py to one of the available setups (HHDM,EFT,GEN,...) inside the directory "setups":
```bash
python setup.py -a HHDM
```

You can check for different cases [**m**= 0(signal), 1-4(bkg all years), 5(data)] if your code is working as intended using the test datasets:

```bash
python runSelection.py -c m
```

Know how many jobs the code is setted to process:

```bash
python runSelection.py -j -1
```

Produce a list of all jobs the code is setted to process:

```bash
python runSelection.py -j -2
```

Produce a list of all missing datsets by the time of the text files creation:

```bash
python runSelection.py -j -3
```

(Only at DESY) Produce a list of all files that exist but are missing at DESY:

```bash
python runSelection.py -j -4
```

Run the job in the **nth** position of the list:

```bash
python runSelection.py -j n
```

# Submiting condor jobs

If you have permission to deploy condor jobs, you can run your code in each dataset as a job.

1. See all flavours available for the jobs
2. Submit all the **N** jobs the code is setted to process (need to provide the proxy)

```bash
./submit_jobs.sh help
./submit_jobs.sh flavour N
```

# Checking and processing condor jobs results

First, go to **tools** directory.

```
cd tools
```

Check integrity of jobs of the selection **Test** and year **2016**:

```
python checker.py -s Test -p 16
```
Check a specific dataset:
```
python checker.py -s Test -p 16 -d TTTo2L2Nu
```

If you want to remove bad jobs, type:

```
python remove_jobs.py -s Test -l <list of bad jobs>
```

Once all jobs are good, you can group them by typing:

```
python grouper.py -s Test -p 16
```

If it is **2016 APV**, type:

```
python checker.py -s Test -p 16 --apv
python grouper.py -s Test -p 16 --apv
```

If your anafile was set to produce systematic histograms, you need to add the syst flag to check and group as well the json files where are stored the histograms. Examples:

```
python checker.py -s Test -p 16 --apv --syst
python grouper.py -s Test -p 16 --apv --syst

python checker.py -s Test -p 18 --syst
python grouper.py -s Test -p 18 --syst
```

By default, checker and grouper use the number of CPUs available minus 2. You can force a specific number. For example, using 5 CPUs:
```
python checker.py -s Test -p 16 -c 5
python grouper.py -s Test -p 16 -c 5
```

If the code is crashing, the debug flag can help you to identify the problematic folder:
```
python checker.py -s Test -p 16 --debug
python grouper.py -s Test -p 16 --debug
```
In the checker, the problematic dataset is known. In order to save time, it is recommended to use the debug flag in combination with the name of the dataset you want to investigate.
```
python checker.py -s Test -p 16 -d TTTo2L2Nu --debug
```

# Resubmiting condor jobs

If there are bad jobs in the output directory, they will be written in the **tools/resubmit_YY.txt** file, where **YY** is the year associated with the job. If you desire to resubmit the bad jobs listed in these files, you can use the flag ""--resubmit"" as in the commands below.

Know how many jobs the code is setted to process in the resubmission:

```bash
python runSelection.py -j -1 --resubmit
```

Produce a list of all jobs the code is setted to process in the resubmission:

```bash
python runSelection.py -j -2 --resubmit
```

Resubmit your jobs:

```bash
./submit_jobs.sh flavour N --resubmit
```


# Working with anafiles

Create a template for a new anafile called **TrigEff** and integrate it to the framework: 

```bash
./addSelection.sh TrigEff
```

Dissociate **TrigEff** from the framework (the anafile is not deleted):

```bash
./rmSelection.sh TrigEff
```


# Local development with Docker

This repository ships a development container and a docker-compose file that aims to simulate the lxplus environment and `CVMFS` in your local machine. In order to use the development container you need to use Visual Studio Code (VSCode) with Remote Development extension.

Simulating `CVMFS` (CERN VM File System) is quite easy using `cvmfs/service` docker image, the `docker-compose.yml` already has the necessary caveats to properly start the container service and store the cache in your local machine (for better perforamnce after first use). Install docker (use the recipe in https://docs.docker.com/engine/install/) and docker-compose, then 
add your user name to the docker group typing:

```bash
sudo usermod -aG docker $USER
```

Using the VSCode terminal synchronized to the directory of the HEPHero, start the detached service with:

```bash
docker-compose up -d
```

When desired, stop the service running:

```bash
docker-compose stop
```

Remove the service from your machine with:

```bash
docker-compose down
```

After installing press `Ctrl + Shift + P` and type `reopen in container` and select the option `Remote-Containers: Reopen in Container`. This will start a session inside the container environment, if it is the first the opening the development container it will take a time to setup.

Note: If cvmfs directories do not load inside devcontainer, rebuilt the container.

hepenv setup:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_100/x86_64-centos7-gcc9-opt/setup.sh; python -m venv hepenv; source hepenv/bin/activate
```
