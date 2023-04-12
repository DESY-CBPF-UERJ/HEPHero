#!/bin/bash

# Check if HEP_OUTPATH variable exists then read
if [[ -z "${HEP_OUTPATH}" ]]; then
  echo "HEP_OUTPATH environment varibale is undefined. Aborting script execution..."
  exit 1
else
  outpath=${HEP_OUTPATH}
fi

# Check if MACHINES variable exists then read
if [[ -z "${MACHINES}" ]]; then
  echo "MACHINES environment varibale is undefined. Aborting script execution..."
  exit 1
else
  machines=${MACHINES}
fi

# Check if REDIRECTOR variable exists then read
if [[ -z "${REDIRECTOR}" ]]; then
  echo "REDIRECTOR environment varibale is undefined. Defaulting to xrootd-cms.infn.it"
  redirector='xrootd-cms.infn.it'
else
  redirector=${REDIRECTOR}
fi

N_datasets=$2
flavor=\"$1\"
resubmit=$3

# Check if third argument is the submission flag
if [ ${resubmit} ] && [ "${resubmit}" != "--resubmit" ]; then
  echo "Submission flag is incorrect."
  exit 1
fi

if [ ${resubmit} ] && [ "${resubmit}" == "--resubmit" ]; then
  run_fix="no"
else
  run_fix="yes"
fi

if [ "$1" == "help" ]
then
    echo "command: ./submit_grid.sh Flavour NumberOfJobs" 
    echo "Options for Flavour (maximum time to complete all jobs):"
    echo "espresso     = 20 minutes"
    echo "microcentury = 1 hour"
    echo "longlunch    = 2 hours"
    echo "workday      = 8 hours"
    echo "tomorrow     = 1 day"
    echo "testmatch    = 3 days"
    echo "nextweek     = 1 week"
else
    #==================================================================================================
    if [ "${machines}" == "CERN" ]; then
        Proxy_file=/afs/cern.ch/user/${USER:0:1}/${USER}/private/x509up
        voms-proxy-init --voms cms
        cp /tmp/x509up_u$(id -u) ${Proxy_file}
    fi
    
    if [ "${machines}" == "DESY" ]; then
        Proxy_file=None
    fi
    
    sed -i "s/.*queue.*/queue ${N_datasets}/" HTCondor/condor.sub
    sed -i "s~.*Proxy_path            =.*~Proxy_path            = ${Proxy_file}~" HTCondor/condor.sub
    sed -i "s~.*arguments.*~arguments             = \$(ProcId) \$(Proxy_path) $(pwd) ${outpath} ${redirector} ${machines} ${resubmit}~" HTCondor/condor.sub
    sed -i "s/.*+JobFlavour.*/+JobFlavour             = ${flavor}/" HTCondor/condor.sub
    
    if [ "${run_fix}" == "yes" ]; then
    python runSelection.py -j 0 --start
    rm HTCondor/*.log
    rm HTCondor/jobs_log/run*
    fi
    
    rm -rf hepenv
    cd HTCondor
    condor_submit condor.sub
fi

#used in submit_jobs.sh
#sed -i "s~.*transfer_input_files.*~transfer_input_files    = $(pwd)~" HTCondor/condor.sub

#used in condor.sub
#should_transfer_files   = YES
#when_to_transfer_output = ON_EXIT
#transfer_input_files    = file1,file2

