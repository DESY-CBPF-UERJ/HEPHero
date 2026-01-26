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


# Check if MACHINES variable exists then read
if [[ -z "${MACHINES}" ]]; then
  echo "MACHINES environment varibale is undefined. Aborting script execution..."
  exit 1
else
  machines=${MACHINES}
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
    cd ..
    tgzdir=$(pwd)
    rm HEPHeroML.tgz
    tar --exclude='HEPHeroML/.git*' --exclude='HEPHeroML/condor' --exclude='HEPHeroML/history' --exclude='HEPHeroML/Example' -zcf HEPHeroML.tgz HEPHeroML
    cd HEPHeroML

    sed -i "s/.*queue.*/queue ${N_models}/" train.sub
    sed -i "s~.*arguments.*~arguments             = \$(ProcId) ${machines} ${trainer}~" train.sub
    sed -i "s/.*+JobFlavour.*/+JobFlavour = ${flavor}/" train.sub
    sed -i "s~.*transfer_input_files.*~transfer_input_files  = ${tgzdir}/HEPHeroML.tgz~" train.sub
    sed -i "s/.*should_transfer_files.*/should_transfer_files = YES/" train.sub

    python $trainer --clean
    condor_submit train.sub
fi












