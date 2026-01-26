**HEPHeroML - Machine Learning tool for the DESY-CBPF-UERJ collaboration**

General information
-----------

* This code is meant to be used in association with the HEPHero framework.

* The training setup is made in the python scripts inside the directory **setups**.

* The input data consists of h5 files created by the tool **grouper.py** of the HEPHero framework.

* The training results and files are stored in the smae directory of the h5 files.


Starting
-----------

Inside your private area (NOT in the eos or dust area and NOT inside a CMSSW release), download the code.  
```bash
git clone https://github.com/DESY-CBPF-UERJ/HEPHeroML.git
```

Source the hepenv environment before work with the HEPHeroML:
```
hepenv
```

Enter in the HEPHeroML directory:  
```bash
cd HEPHeroML
```


Generating the trainer
-----------

After setup the model and training in one of the python scripts inside **setups**, generate the trainer using the **generate_trainer.py** script. Example: Generate the trainer for the analysis **OPENDATA** with the tag **Class** (defined inside **setups/OPENDATA.py**):
```bash
python generate_trainer.py -a OPENDATA
```
It will create the trainer script **train_OPENDATA_Class.py**.



Running the trainer
-----------
Know how many jobs the code is setted to train (information needed to submit jobs):
```bash
python train_OPENDATA_Class.py -j -1
```

List the jobs the code is setted to train:
```bash
python train_OPENDATA_Class.py -j -2
```

Train the model in the position **n** of the list:
```bash
python train_OPENDATA_Class.py -j n
```
Ex.:
```bash
python train_OPENDATA_Class.py -j 2
```

Submit condor jobs
-----------
1. Make **submit_jobs.sh** an executable:  
```bash
chmod +x submit_jobs.sh
```   
2. See all flavours available for the jobs:  
```bash
./submit_jobs.sh -h
```  
3. Submit all the **N** jobs the code is set to train:  
```bash
./submit_jobs.sh -f flavour -n N -t trainer
```  
Ex.:
```bash
./submit_jobs.sh -f workday -n 32 -t train_OPENDATA_Class.py
```

Evaluate the results
-----------
After the jobs have finished, evaluate the training results:
```bash
python train_OPENDATA_Class.py --evaluate
```



