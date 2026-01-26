#!/bin/bash

echo "ProcId"
echo $1
echo "MACHINES"
echo $2
echo "TRAINER"
echo $3

export MACHINES=$2
ls

if [ "$2" == "CERN" ]; then
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
elif [ "$2" == "UERJ" ]; then
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el8-gcc11-opt/setup.sh
fi

cd HEPHeroML 
python $3 -j $1
