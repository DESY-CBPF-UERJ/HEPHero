#!/bin/bash

echo "ProcId"
echo $1
echo "MACHINES"
echo $2
echo "STORAGE_REDIRECTOR"
echo $3
echo "STORAGE_USER"
echo $4
echo "TRAINER"
echo $5

export MACHINES=$2
export STORAGE_REDIRECTOR=$3
export STORAGE_USER=$4
ls

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el8-gcc11-opt/setup.sh

#if [ "$2" == "CERN" ]; then
#source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
#elif [ "$2" == "UERJ" ]; then
#source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el8-gcc11-opt/setup.sh
#fi

if [ "$STORAGE_REDIRECTOR" == "eosuser.cern.ch" ]; then
    STORAGE_DIR=eos/user/${STORAGE_USER:0:1}/${STORAGE_USER}
elif [ "$STORAGE_REDIRECTOR" == "xrootd2.hepgrid.uerj.br:1094" ]; then
    STORAGE_DIR=store/user/${STORAGE_USER}
fi

if [ "$STORAGE_REDIRECTOR" != "None" ]; then
mkdir output
export HEP_OUTPATH=$(pwd)/output
fi

voms-proxy-info -all -file ${X509_USER_PROXY}

tar -zxf ML.tgz
cd ML 
python $5 -j $1 --condor
cd ..

if [ "$STORAGE_REDIRECTOR" != "None" ]; then
echo root://$STORAGE_REDIRECTOR//${STORAGE_DIR}
root --version

xrdcp -rf output root://$STORAGE_REDIRECTOR//${STORAGE_DIR}
fi



