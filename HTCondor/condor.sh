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
echo "USER"
echo $7
echo "ANALYSIS"
echo $8
echo "RESUBMISSION"
echo $9


export HEP_OUTPATH=$4
export REDIRECTOR=$5
export MACHINES=$6
export USER=$7

if [ "$6" == "CERN" ]; then
rsync -azh --exclude="HEPHero/.*" --exclude="HEPHero/CMakeFiles" --exclude="HEPHero/RunAnalysis" --exclude="HEPHero/HTCondor/*.log" --exclude="HEPHero/HTCondor/jobs_log/run_*" --exclude="HEPHero/AP_*" $3 .
cd HEPHero
rsync -azh --exclude="$8/Datasets/*.hepmc" --exclude="$8/Datasets/*.root" --exclude="$8/ana/local_output" $3/$8 .
cd ..
#=======
#cp -rf $HEP_OUTPATH/HEPHero .
#=======
#rsync -azh $HEP_OUTPATH/HEPHero .
#=======
#git clone https://github.com/DESY-CBPF-UERJ/HEPHero.git
#cd HEPHero
#cp ana/templates/Test.cpp ana
#./addSelection.sh Test
#cd ..
#=======
export MY_TORCH_PATH=/afs/cern.ch/user/${USER:0:1}/${USER}/libtorch
export MY_ONNX_PATH=/afs/cern.ch/user/${USER:0:1}/${USER}/onnxruntime-linux-x64-1.20.1
fi    

if [ "$6" == "CMSC" ]; then
export $2=$(pwd)/$2
mkdir output
cd output
export HEP_OUTPATH=$(pwd)
cd ..
tar -zxf HEPHero.tgz
rm HEPHero.tgz
#xrdcp -fr HEPHero.tgz root://cmsxrootd.fnal.gov///store/user/gcorreia
mv HEPHero HEPHero_old
mkdir HEPHero
mv HEPHero_old/* HEPHero
rm -rf HEPHero_old
xrdcp -fr root://cmsxrootd.fnal.gov///store/user/gcorreia/libtorch.tgz .
xrdcp -fr root://cmsxrootd.fnal.gov///store/user/gcorreia/onnxruntime-linux-x64-1.20.1.tgz .
tar -zxf libtorch.tgz
tar -zxf onnxruntime-linux-x64-1.20.1.tgz
rm libtorch.tgz
rm onnxruntime-linux-x64-1.20.1.tgz
export MY_TORCH_PATH=$(pwd)/libtorch
export MY_ONNX_PATH=$(pwd)/onnxruntime-linux-x64-1.20.1
fi
    
if [ "$6" == "DESY" ]; then
cd ../..
export MY_TORCH_PATH=/afs/desy.de/user/${USER:0:1}/${USER}/libtorch
source /cvmfs/grid.desy.de/etc/profile.d/grid-ui-env.sh
fi

if [ "$6" == "UERJ" ]; then
mkdir output
cd output
export HEP_OUTPATH=$(pwd)
cd ..
tar zxf HEPHero.tgz
mv HEPHero HEPHero_old
mkdir HEPHero
mv HEPHero_old/* HEPHero
rm -rf HEPHero_old
export MY_TORCH_PATH=/mnt/hadoop/cms/store/user/${USER}/libtorch
fi

#if [ "$6" == "CERN" ] || [ "$6" == "DESY" ]; then
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
#else
#source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-centos7-gcc11-opt/setup.sh
#fi
cd HEPHero

if [ "$6" != "DESY" ]; then
rm CMakeCache.txt
cmake $(pwd)
make -j 4
fi

if [ "$9" == "--resubmit" ]; then
python runSelection.py -j $1 -p $2 -t 0 $9
else
python runSelection.py -j $1 -p $2 -t 0
fi


if [ "$6" == "CERN" ]; then
cd ..
sed -i '/Error in <TNetXNGFile::TNetXNGFile>: The remote file is not open/d' _condor_stderr
fi


