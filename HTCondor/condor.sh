#!/bin/bash

echo "ProcId"
echo $1
echo "Proxy_path"
echo $2
echo "HEPHero_path"
echo $3
echo "HEP_OUTPATH"
echo $4
echo "REDIRECTOR"
echo $5
echo "MACHINES"
echo $6
echo "RESUBMISSION"
echo $7

export HEP_OUTPATH=$4
export REDIRECTOR=$5
export MACHINES=$6

if [ "$6" == "CERN" ]; then
rsync -azh --exclude="HEPHero/.*" --exclude="HEPHero/CMakeFiles" --exclude="HEPHero/RunAnalysis" --exclude="HEPHero/Datasets/*.root" --exclude="HEPHero/HTCondor/*.log" --exclude="HEPHero/HTCondor/jobs_log/run_*" --exclude="HEPHero/ana/local_output" $3 .
export MY_TORCH_PATH=/afs/cern.ch/user/${USER:0:1}/${USER}/libtorch
fi    
    
if [ "$6" == "DESY" ]; then
cd ../..
export MY_TORCH_PATH=/afs/desy.de/user/${USER:0:1}/${USER}/libtorch
fi

ls
source /cvmfs/sft.cern.ch/lcg/views/LCG_100/x86_64-centos7-gcc9-opt/setup.sh
python -m venv hepenv
source hepenv/bin/activate
cd HEPHero

if [ "$6" == "CERN" ]; then
rm CMakeCache.txt
cmake .
make -j 4
fi

if [ "$7" == "--resubmit" ]; then
python runSelection.py -j $1 -p $2 -t 0 $7
else
python runSelection.py -j $1 -p $2 -t 0 
fi

#echo ${SELECTION}
#echo ${JOB_NAME}
#cd tools
#python checker.py -s ${SELECTION} -n ${JOB_NAME}


if [ "$6" == "CERN" ]; then
cd ..
sed -i '/Error in <TNetXNGFile::TNetXNGFile>: The remote file is not open/d' _condor_stderr
fi


