#!/bin/bash
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Syntax: ./submit_jobs.sh [-h|r|s|l] -f job_flavour -n number_of_jobs"
   echo "options:"
   echo "h     Print this Help."
   echo "r     Resubmit failed jobs."
   echo "s     Store output in /store/user/<username>."
   echo "l     It will run the jobs locally."
   echo "f     Job flavour."
   echo "n     Number of jobs."
   echo
   echo "Options for job_flavour (maximum time to complete all jobs):"
   echo "espresso     = 20 minutes"
   echo "microcentury = 1 hour"
   echo "longlunch    = 2 hours"
   echo "workday      = 8 hours"
   echo "tomorrow     = 1 day"
   echo "testmatch    = 3 days"
   echo "nextweek     = 1 week"
}
############################################################
# Process the input options. Add options as needed.        #
############################################################
# Get the options
while getopts ":hrsl:f:n:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
     r) # resubmit jobs
         resubmit=yes
         echo "Resubmission mode activated.";;
     s) # store output in "/store/user/<username>"
         storage=yes;;
     l) # run the jobs locally
         local=yes;;
     f) # job flavour
         flavour=\"$OPTARG\";;
     n) # number of jobs
         N_datasets=$OPTARG;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

voms-proxy-init --voms cms

# Check if MACHINES variable exists then read
if [[ -z "${MACHINES}" ]]; then
  echo "MACHINES environment varibale is undefined. Aborting script execution..."
  exit 1
else
  machines=${MACHINES}
fi


if [ "${machines}" == "UERJ" ] || [ "${machines}" == "CMSC" ]; then
storage=yes
fi


if [ $storage ] && [ "$storage" == "yes" ]; then
  echo "The output will be stored in the user storage."
  storage_redirector=${STORAGE_REDIRECTOR}
  storage_user=${STORAGE_USER}
  if [ "$STORAGE_REDIRECTOR" == "eosuser.cern.ch" ]; then
      storage_dir=eos/user/${STORAGE_USER:0:1}/${STORAGE_USER}
  elif [ "$STORAGE_REDIRECTOR" == "xrootd2.hepgrid.uerj.br:1094" ]; then
      storage_dir=store/user/${STORAGE_USER}
  fi
else
  echo "The output will be stored at ${HEP_OUTPATH}."
  storage_redirector=None
  storage_user=None
  storage_dir=None
fi


# Check if HEP_OUTPATH variable exists then read
if [[ -z "${HEP_OUTPATH}" ]]; then
  echo "HEP_OUTPATH environment varibale is undefined. Aborting script execution..."
  exit 1
else
  outpath=${HEP_OUTPATH}
fi


# Check if REDIRECTOR variable exists then read
if [[ -z "${REDIRECTOR}" ]]; then
  echo "REDIRECTOR environment varibale is undefined. Defaulting to xrootd-cms.infn.it"
  redirector='xrootd-cms.infn.it'
else
  redirector=${REDIRECTOR}
fi


ANALYSIS=$(<tools/analysis.txt)

if [ ${resubmit} ] && [ "${resubmit}" == "yes" ]; then
  run_start="no"
else
  run_start="yes"
fi


if [ $local ] && [ "$local" == "yes" ]; then
    python runSelection.py -j 0 --start
    ijob=0
    while (( $ijob < $2 ))
    do
      python runSelection.py -j $ijob
      ijob=$(( ijob+1 ))
    done
else
    cd ..
    tgzdir=$(pwd)
    rm HEPHero.tgz
    rm AP.tgz
    tar --exclude='HEPHero/RunAnalysis' --exclude='HEPHero/.git*' --exclude='HEPHero/HTCondor/*.log' --exclude='HEPHero/HTCondor/jobs_log/run_*' --exclude='HEPHero/CMakeFiles' --exclude='HEPHero/AP_*' -zcf HEPHero.tgz HEPHero
    tar --exclude="HEPHero/${ANALYSIS}/.git*" --exclude="HEPHero/${ANALYSIS}/Datasets/*.hepmc" --exclude="HEPHero/${ANALYSIS}/Datasets/*.root" --exclude="HEPHero/${ANALYSIS}/ana/local_output" -zcf AP.tgz HEPHero/${ANALYSIS}
    cd HEPHero

    if [ "${machines}" == "CERN" ]; then
        Proxy_filename=x509up
        cp /tmp/x509up_u$(id -u) /afs/cern.ch/user/${USER:0:1}/${USER}/private/x509up
        sed -i "s/.*Universe.*/#Universe              =/" HTCondor/condor.sub
        sed -i "s~.*x509userproxy = /tmp.*~#x509userproxy = /tmp/${Proxy_filename}~" HTCondor/condor.sub
        sed -i "s/.*use_x509userproxy.*/#use_x509userproxy = true/" HTCondor/condor.sub
        sed -i "s~.*+REQUIRED_OS.*~#+REQUIRED_OS           =~" HTCondor/condor.sub
        sed -i "s~.*request_cpus.*~#request_cpus           =~" HTCondor/condor.sub
        sed -i "s/.*accounting_group_user.*/#accounting_group_user =/" HTCondor/condor.sub
        sed -i "s/.*accounting_group      =.*/#accounting_group      =/" HTCondor/condor.sub
        sed -i "s/.*grid_resource.*/#grid_resource         =/" HTCondor/condor.sub
    elif [ "${machines}" == "CMSC" ]; then
        Proxy_filename=x509up_u$(id -u)
        sed -i "s/.*Universe.*/Universe              = vanilla/" HTCondor/condor.sub
        sed -i "s~.*x509userproxy = /tmp.*~#x509userproxy = /tmp/${Proxy_filename}~" HTCondor/condor.sub
        sed -i "s/.*use_x509userproxy.*/#use_x509userproxy = true/" HTCondor/condor.sub
        sed -i "s~.*+REQUIRED_OS.*~+REQUIRED_OS           = \"rhel9\"~" HTCondor/condor.sub
        sed -i "s~.*request_cpus.*~request_cpus           = 2~" HTCondor/condor.sub
        sed -i "s/.*accounting_group_user.*/#accounting_group_user =/" HTCondor/condor.sub
        sed -i "s/.*accounting_group      =.*/#accounting_group      =/" HTCondor/condor.sub
        sed -i "s/.*grid_resource.*/#grid_resource         =/" HTCondor/condor.sub
    elif [ "${machines}" == "UERJ" ]; then
        Proxy_filename=x509up_u$(id -u)
        sed -i "s/.*Universe.*/Universe              = grid/" HTCondor/condor.sub
        sed -i "s~.*x509userproxy = /tmp.*~x509userproxy = /tmp/${Proxy_filename}~" HTCondor/condor.sub
        sed -i "s/.*use_x509userproxy.*/use_x509userproxy = true/" HTCondor/condor.sub
        sed -i "s~.*+REQUIRED_OS.*~#+REQUIRED_OS           =~" HTCondor/condor.sub
        sed -i "s~.*request_cpus.*~#request_cpus           =~" HTCondor/condor.sub
        sed -i "s/.*accounting_group_user.*/accounting_group_user = ${USER}/" HTCondor/condor.sub
        sed -i "s/.*accounting_group      =.*/accounting_group      = group_uerj/" HTCondor/condor.sub
        sed -i "s/.*grid_resource.*/grid_resource         = condor condor-manager2.hepgrid.uerj.br condor-manager2.hepgrid.uerj.br/" HTCondor/condor.sub
    fi

    sed -i "s~.*transfer_input_files.*~transfer_input_files  = ${tgzdir}/HEPHero.tgz,${tgzdir}/AP.tgz~" HTCondor/condor.sub
    sed -i "s/.*should_transfer_files.*/should_transfer_files = YES/" HTCondor/condor.sub
    sed -i "s~.*arguments.*~arguments             = \$(ProcId) ${Proxy_filename} $(pwd) ${outpath} ${redirector} ${machines} ${USER} ${ANALYSIS} ${storage_redirector} ${storage_user} ${resubmit}~" HTCondor/condor.sub
    #sed -i "s/.*Requirements.*/Requirements            = (HasSingularity == True)/" HTCondor/condor.sub
    #sed -i "s~.*+SingularityImage.*~+SingularityImage       = \"/cvmfs/singularity.opensciencegrid.org/opensciencegrid/osgvo-el8:latest\"~" HTCondor/condor.sub
    #sed -i "s/.*when_to_transfer_output.*/when_to_transfer_output = ON_EXIT/" HTCondor/condor.sub
    #sed -i "s/.*transfer_output_files.*/#transfer_output_files = output/" HTCondor/condor.sub
    #sed -i "s~.*output_destination.*~#output_destination = root://xrootd2.hepgrid.uerj.br:1094//store/user/gcorreia/~" HTCondor/condor.sub
    #sed -i "s~.*transfer_output_remaps.*~transfer_output_remaps = \"output = /home/${USER}/output\"~" HTCondor/condor.sub
    sed -i "s/.*queue.*/queue ${N_datasets}/" HTCondor/condor.sub
    sed -i "s/.*+JobFlavour.*/+JobFlavour             = ${flavour}/" HTCondor/condor.sub


    if [ "${run_start}" == "yes" ]; then
    python runSelection.py -j 0 --start
    rm HTCondor/*.log
    rm HTCondor/jobs_log/run*
    fi

    if [ $storage ] && [ "$storage" == "yes" ]; then
    SELECTION=$(<tools/selection.txt)
    #rm -rf ${outpath}/${ANALYSIS}/${SELECTION}/*_files_*
    xrdfs root://${storage_redirector}/ mkdir /${storage_dir}/output
    xrdfs root://${storage_redirector}/ mkdir /${storage_dir}/output/${ANALYSIS}
    xrdfs root://${storage_redirector}/ mkdir /${storage_dir}/output/${ANALYSIS}/${SELECTION}
    xrdcp -rf ${outpath}/${ANALYSIS}/${SELECTION}/jobs.txt root://${storage_redirector}//${storage_dir}/output/${ANALYSIS}/${SELECTION}
    xrdcp -rf ${outpath}/${ANALYSIS}/${SELECTION}/hephero_local.json root://${storage_redirector}//${storage_dir}/output/${ANALYSIS}/${SELECTION}
    xrdcp -rf ${outpath}/${ANALYSIS}/${SELECTION}/lateral_systematics.json root://${storage_redirector}//${storage_dir}/output/${ANALYSIS}/${SELECTION}
    xrdcp -rf ${outpath}/${ANALYSIS}/${SELECTION}/vertical_systematics.json root://${storage_redirector}//${storage_dir}/output/${ANALYSIS}/${SELECTION}
    xrdcp -rf ${outpath}/${ANALYSIS}/${SELECTION}/${SELECTION}.cpp root://${storage_redirector}//${storage_dir}/output/${ANALYSIS}/${SELECTION}
    fi
    
    cd HTCondor
    condor_submit condor.sub
fi



