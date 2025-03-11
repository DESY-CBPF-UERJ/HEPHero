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

ANALYSIS=$(<tools/analysis.txt)

# Check if third argument is the submission flag
if [ ${resubmit} ] && [ "${resubmit}" != "--resubmit" ]; then
  echo "Submission flag is incorrect."
  exit 1
fi

if [ ${resubmit} ] && [ "${resubmit}" == "--resubmit" ]; then
  run_start="no"
else
  run_start="yes"
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
    echo ""
    echo "local        = it will run the jobs locally"
elif [ "$1" == "local" ]
then
    python runSelection.py -j 0 --start
    ijob=0
    while (( $ijob < $2 ))
    do
      python runSelection.py -j $ijob
      ijob=$(( ijob+1 ))
    done
else
    #==================================================================================================
    if [ "${machines}" == "CERN" ]; then
        Proxy_file=/afs/cern.ch/user/${USER:0:1}/${USER}/private/x509up
        cp /tmp/x509up_u$(id -u) ${Proxy_file}
        #rsync -azh --exclude="HEPHero/.*" --exclude="HEPHero/CMakeFiles" --exclude="HEPHero/RunAnalysis" --exclude="HEPHero/Datasets/*.hepmc" --exclude="HEPHero/Datasets/*.root" --exclude="HEPHero/HTCondor/*.log" --exclude="HEPHero/HTCondor/jobs_log/run_*" --exclude="HEPHero/ana/local_output" $(pwd) ${HEP_OUTPATH}
    fi

    if [ "${machines}" == "CMSC" ]; then
        Proxy_file=x509up_u$(id -u)
        rsync -azh --exclude=".*" --exclude="CMakeFiles" --exclude="RunAnalysis" --exclude="HTCondor/*.log" --exclude="HTCondor/jobs_log/run_*" --exclude="AP_*" $(pwd) ${outpath}
        rsync -azh --exclude="$8/Datasets/*.hepmc" --exclude="$8/Datasets/*.root" --exclude="$8/ana/local_output" $(pwd)/${ANALYSIS} ${outpath}/HEPHero
        hephero_path=$PWD
        cd ${outpath}
        tar -zcf HEPHero.tgz -C $PWD HEPHero
        cd ${hephero_path}
        rm -rf ${outpath}/HEPHero
    fi
    
    if [ "${machines}" == "DESY" ]; then
        Proxy_file=None
        #source /cvmfs/grid.desy.de/etc/profile.d/grid-ui-env.sh
        #Proxy_file=/afs/desy.de/user/${USER:0:1}/${USER}/private/x509up
        #cp /tmp/x509up_u$(id -u) ${Proxy_file}
    fi
    
    if [ "${machines}" == "UERJ" ]; then
        Proxy_file=None
        cd ..
        tgzdir=$(pwd)
        rm HEPHero.tgz
        rm AP.tgz

        tar --exclude='HEPHero/RunAnalysis' --exclude='HEPHero/HTCondor/*.log' --exclude='HEPHero/HTCondor/jobs_log/run_*' --exclude='HEPHero/CMakeFiles' --exclude='HEPHero/AP_*' -zcf HEPHero.tgz HEPHero

        tar --exclude="HEPHero/${ANALYSIS}/Datasets/*.hepmc" --exclude="HEPHero/${ANALYSIS}/Datasets/*.root" --exclude="HEPHero/${ANALYSIS}/ana/local_output" -zcf AP.tgz HEPHero/${ANALYSIS}

        cd HEPHero
        if grep -q "#grid_resource" HTCondor/condor.sub; then
            sed -i "s/.*Universe.*/Universe              = grid/" HTCondor/condor.sub
            sed -i "s/.*accounting_group_user.*/accounting_group_user = ${USER}/" HTCondor/condor.sub
            sed -i "s/.*accounting_group      =.*/accounting_group      = group_uerj/" HTCondor/condor.sub
            sed -i "s/.*grid_resource.*/grid_resource         = condor condor.hepgrid.uerj.br condor.hepgrid.uerj.br/" HTCondor/condor.sub
            sed -i "s~.*transfer_input_files.*~transfer_input_files  = ${tgzdir}/HEPHero.tgz,${tgzdir}/AP.tgz~" HTCondor/condor.sub
            sed -i "s/.*should_transfer_files.*/should_transfer_files = YES/" HTCondor/condor.sub
            sed -i "s/.*when_to_transfer_output.*/when_to_transfer_output = ON_EXIT/" HTCondor/condor.sub
            sed -i "s/.*transfer_output_files.*/transfer_output_files = output/" HTCondor/condor.sub
            sed -i "s~.*transfer_output_remaps.*~transfer_output_remaps = \"output = /home/${USER}/output\"~" HTCondor/condor.sub
        fi
    elif [ "${machines}" == "CMSC" ]; then
        if grep -q "#grid_resource" HTCondor/condor.sub; then
            sed -i "s/.*Universe.*/Universe              = vanilla/" HTCondor/condor.sub
            sed -i "s~.*transfer_input_files.*~transfer_input_files  = ${outpath}/HEPHero.tgz~" HTCondor/condor.sub
            sed -i "s/.*should_transfer_files.*/should_transfer_files = YES/" HTCondor/condor.sub
            sed -i "s/.*when_to_transfer_output.*/when_to_transfer_output = ON_EXIT/" HTCondor/condor.sub
            sed -i "s/.*transfer_output_files.*/transfer_output_files = output/" HTCondor/condor.sub
            sed -i "s~.*transfer_output_remaps.*~transfer_output_remaps = \"output = ${outpath}\"~" HTCondor/condor.sub
            sed -i "s~.*+REQUIRED_OS.*~+REQUIRED_OS           = \"rhel9\"~" HTCondor/condor.sub
            sed -i "s~.*request_cpus.*~request_cpus           = 4~" HTCondor/condor.sub
        fi
    else
        if ! grep -q "#grid_resource" HTCondor/condor.sub; then
            sed -i "s/.*Universe.*/#Universe              = grid/" HTCondor/condor.sub
            sed -i "s/.*accounting_group_user.*/#accounting_group_user = ${USER}/" HTCondor/condor.sub
            sed -i "s/.*accounting_group      =.*/#accounting_group      = group_uerj/" HTCondor/condor.sub
            sed -i "s/.*grid_resource.*/#grid_resource         = condor condor.hepgrid.uerj.br condor.hepgrid.uerj.br/" HTCondor/condor.sub
            sed -i "s~.*transfer_input_files.*~#transfer_input_files  = ${tgzdir}/HEPHero.tgz,${tgzdir}/AP.tgz~" HTCondor/condor.sub
            sed -i "s/.*should_transfer_files.*/#should_transfer_files = YES/" HTCondor/condor.sub
            sed -i "s/.*when_to_transfer_output.*/#when_to_transfer_output = ON_EXIT/" HTCondor/condor.sub
            sed -i "s/.*transfer_output_files.*/#transfer_output_files = output/" HTCondor/condor.sub
            sed -i "s~.*transfer_output_remaps.*~#transfer_output_remaps = \"output = /home/${USER}/output\"~" HTCondor/condor.sub
        fi
    fi
    
    sed -i "s~.*Proxy_path            .*~Proxy_path            = ${Proxy_file}~" HTCondor/condor.sub
    sed -i "s/.*queue.*/queue ${N_datasets}/" HTCondor/condor.sub
    sed -i "s~.*arguments.*~arguments             = \$(ProcId) \$(Proxy_path) $(pwd) ${outpath} ${redirector} ${machines} ${USER} ${ANALYSIS} ${resubmit}~" HTCondor/condor.sub
    sed -i "s/.*+JobFlavour.*/+JobFlavour             = ${flavor}/" HTCondor/condor.sub
    
    if [ "${run_start}" == "yes" ]; then
    python runSelection.py -j 0 --start
    rm HTCondor/*.log
    rm HTCondor/jobs_log/run*
    fi
    
    cd HTCondor
    condor_submit condor.sub
fi



