#!/bin/bash
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Syntax: ./submit_jobs.sh [-h|l] -f job_flavour -n number_of_jobs -t trainer"
   echo "options:"
   echo "h     Print this Help."
   echo "l     It will run the jobs locally."
   echo "f     Job flavour."
   echo "n     Number of models to train (jobs)."
   echo "t     trainer (python file with training setup)."
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
while getopts ":hlk:f:n:t:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
     l) # run the jobs locally
         local=yes;;
     f) # job flavour
         flavour=\"$OPTARG\";;
     n) # number of jobs
         N_models=$OPTARG;;
     t) # trainer
         trainer=$OPTARG;;
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

if [ $local ] && [ "$local" == "yes" ]; then
  echo "The output will be stored at ${HEP_OUTPATH}."
  storage_redirector=None
  storage_user=None
else
  echo "The output will be stored in the user storage."
  storage_redirector=${STORAGE_REDIRECTOR}
  storage_user=${STORAGE_USER}
fi


if [ $local ] && [ "$local" == "yes" ]; then
    python $trainer --clean
    ijob=0
    while (( $ijob < $N_models ))
    do
      python $trainer -j $ijob
      ijob=$(( ijob+1 ))
    done
else
    cp ${trainer} ../../ML
    cd ../..
    tgzdir=$(pwd)
    tar --exclude='ML/condor' --exclude='ML/examples' -zcf ML.tgz ML
    cd ML

    Proxy_filename=x509up_u$(id -u)
    sed -i "s/.*Universe.*/Universe              = grid/" train.sub
    sed -i "s~.*x509userproxy = /tmp.*~x509userproxy = /tmp/${Proxy_filename}~" train.sub
    sed -i "s/.*use_x509userproxy.*/use_x509userproxy = true/" train.sub
    sed -i "s/.*accounting_group_user.*/accounting_group_user = ${USER}/" train.sub
    sed -i "s/.*accounting_group      =.*/accounting_group      = group_uerj/" train.sub
    sed -i "s/.*grid_resource.*/grid_resource         = condor condor-manager2.hepgrid.uerj.br condor-manager2.hepgrid.uerj.br/" train.sub
    sed -i "s/.*queue.*/queue ${N_models}/" train.sub
    sed -i "s~.*arguments.*~arguments             = \$(ProcId) ${machines} ${storage_redirector} ${storage_user} ${trainer}~" train.sub
    sed -i "s/.*+JobFlavour.*/+JobFlavour = ${flavor}/" train.sub
    sed -i "s~.*transfer_input_files.*~transfer_input_files  = ${tgzdir}/ML.tgz~" train.sub
    sed -i "s/.*should_transfer_files.*/should_transfer_files = YES/" train.sub

    python $trainer --clean
    condor_submit train.sub

    rm ${trainer}
    cd ..
    rm ML.tgz
fi

