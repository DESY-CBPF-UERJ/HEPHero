**HEPHeroML - Machine Learning tool for the DESY-CBPF-UERJ collaboration**

General information
-----------

* This code is part of the HEPHero project.

* The input data consists of h5 files created by the tool **grouper.py** of the HEPHero framework.


Generating the trainer
-----------

To generate the machine learning trainer using the **generate_trainer.py** script, you must be inside the HEPHero directory and follow the example below. 
Example: Your analysis project is called **AP_Test_OD** and you will generate a trainer using the model **NN** with the tag **Test**:
```bash
python ML/generate_trainer.py -m NN -t Test
```
It will create the trainer script **train_NN_Test.py** inside the directory **AP_Test_OD/ML**. The files **condor.py** and **submit_jobs.sh** will be created inside the same directory.



Running the trainer
-----------
To run the trainer, go to the ML directory of your project.
```bash
cd AP_Test_OD/ML
```

Know how many jobs the code is set to train (information needed to submit jobs):
```bash
python train_NN_Test.py -j -1
```

List the jobs the code is set to train:
```bash
python train_NN_Test.py -j -2
```

Train the model in the position **n** of the list:
```bash
python train_NN_Test.py -j n
```
Ex.:
```bash
python train_NN_Test.py -j 2
```

Submit Condor jobs
-----------
1. Make **submit_jobs.sh** an executable:  
```bash
chmod +x submit_jobs.sh
```   
2. See all flavours available for the jobs:  
```bash
./submit_jobs.sh -h
```  
3. Submit locally all the **N** jobs the code is set to train:  
```bash
./submit_jobs.sh -l -f flavour -n N -t trainer
```  
Ex.:
```bash
./submit_jobs.sh -l -f workday -n 32 -t train_NN_Test.py
```

3. Submit condor jobs for all the **N** jobs the code is set to train:  
```bash
./submit_jobs.sh -f flavour -n N -t trainer
```  
Ex.:
```bash
./submit_jobs.sh -f workday -n 32 -t train_NN_Test.py
```

Evaluate the results
-----------
After the local jobs have finished, evaluate the training results:
```bash
python train_NN_Test.py --evaluate
```

After the condor jobs have finished, evaluate the training results:
```bash
python train_NN_Test.py --evaluate --condor
```



