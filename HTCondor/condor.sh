#!/bin/bash

echo "ProcId"
echo $1
echo "Proxy_filename"
echo $2
echo "HEPHero_path"
echo $3
echo "HEP_OUTPATH"
echo $4
echo "REDIRECTOR"
echo $5
echo "MACHINES"
echo $6
echo "USER"
echo $7
echo "ANALYSIS"
echo $8
echo "STORAGE_REDIRECTOR"
echo $9
echo "STORAGE_USER"
echo ${10}
echo "RESUBMISSION"
echo ${11}

export HEP_OUTPATH=$4
export REDIRECTOR=$5
export STORAGE_REDIRECTOR=$9
export STORAGE_USER=${10}
export MACHINES=$6
export USER=$7

if [ "$STORAGE_REDIRECTOR" == "eosuser.cern.ch" ]; then
    STORAGE_DIR=eos/user/${STORAGE_USER:0:1}/${STORAGE_USER}
elif [ "$STORAGE_REDIRECTOR" == "xrootd2.hepgrid.uerj.br:1094" ]; then
    STORAGE_DIR=store/user/${STORAGE_USER}
fi

if [ "$6" == "CERN" ]; then
export X509_USER_PROXY=/afs/cern.ch/user/${USER:0:1}/${USER}/private/$2
cp /afs/cern.ch/user/${USER:0:1}/${USER}/private/$2 .
#export MY_TORCH_PATH=/afs/cern.ch/user/${USER:0:1}/${USER}/libtorch
export MY_ONNX_PATH=/afs/cern.ch/user/${USER:0:1}/${USER}/onnxruntime-linux-x64-1.20.1
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
elif [ "$6" == "CMSC" ]; then
export $2=$(pwd)/$2
#xrdcp -fr root://${STORAGE_REDIRECTOR}//${STORAGE_DIR}/libtorch-shared-with-deps-1.9.0+cpu.zip .
#xrdcp -fr root://${STORAGE_REDIRECTOR}//${STORAGE_DIR}/fixtorch .
xrdcp -fr root://${STORAGE_REDIRECTOR}//${STORAGE_DIR}/onnxruntime-linux-x64-1.20.1.tgz .
#unzip -o libtorch-shared-with-deps-1.9.0+cpu.zip
tar -zxf onnxruntime-linux-x64-1.20.1.tgz
#rm libtorch-shared-with-deps-1.9.0+cpu.zip
rm onnxruntime-linux-x64-1.20.1.tgz
#cp fixtorch/DeviceGuard.h libtorch/include/ATen
#cp fixtorch/Functions.h libtorch/include/ATen
#export MY_TORCH_PATH=$(pwd)/libtorch
export MY_ONNX_PATH=$(pwd)/onnxruntime-linux-x64-1.20.1
echo "ls"
ls
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
elif [ "$6" == "UERJ" ]; then
#export MY_TORCH_PATH=/cms/store/user/${USER}/libtorch
export MY_ONNX_PATH=/cms/store/user/${USER}/onnxruntime-linux-x64-1.20.1
echo "ls"
ls
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el8-gcc11-opt/setup.sh
fi

voms-proxy-info -all -file ${X509_USER_PROXY}

tar -zxf HEPHero.tgz
tar -zxf AP.tgz
rm HEPHero.tgz
rm AP.tgz
mv HEPHero HEPHero_old
mkdir HEPHero
mv HEPHero_old/* HEPHero
rm -rf HEPHero_old

if [ "$STORAGE_REDIRECTOR" != "None" ]; then
mkdir output
export HEP_OUTPATH=$(pwd)/output
cd HEPHero
  if [ "${11}" == "yes" ]; then
  python runSelection.py -j $1 --start_storage --resubmit
  else
  python runSelection.py -j $1 --start_storage
  fi
cd ..
fi

cd HEPHero
rm CMakeCache.txt
cmake $(pwd)
if [ "$6" == "UERJ" ]; then
make -j 1
else
make -j 2
fi


if [ "${11}" == "yes" ]; then
python runSelection.py -j $1 -t 0 --resubmit
else
python runSelection.py -j $1 -t 0
fi
cd ..

if [ "$STORAGE_REDIRECTOR" != "None" ]; then
  if [ "$6" == "CERN" ]; then
  export X509_USER_PROXY=$2
  voms-proxy-info -all -file ${X509_USER_PROXY}
  fi
xrdcp -rf output root://$STORAGE_REDIRECTOR//${STORAGE_DIR}
fi

if [ "$6" == "CERN" ]; then
sed -i '/Error in <TNetXNGFile::TNetXNGFile>: The remote file is not open/d' _condor_stderr
fi


