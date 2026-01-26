import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", dest='verbose_flag', action='store_true')
parser.set_defaults(verbose_flag=False)
args = parser.parse_args()

list_basedir = os.listdir("../../ML/condor")
log_file = [i for i in list_basedir if ".log" in i][0]
cluster_id = log_file[4:-4]
print("cluster_id = " + cluster_id)

log_file = "../../ML/condor/" + log_file

count_submitted = 0
count_terminated = 0
count_aborted = 0

with open(log_file) as f:
    for line in f:
        count_submitted += line.count("Job submitted from host")
        count_terminated += line.count("Job terminated.")
        count_aborted += line.count("Job was aborted")
        
print('\nJobs submitted:  ' + str(count_submitted))
print('Jobs terminated: ' + str(count_terminated))
print('Jobs aborted:    ' + str(count_aborted) + '\n')

"""
if args.verbose_flag:
    for idxJob in range(count_submitted):
        err_file = "../HTCondor/jobs_log/run_" + str(cluster_id) + "_" + str(idxJob) + ".err"
        if os.path.isfile(err_file):
            count_error = 0
            with open(err_file) as f:
                for line in f:
                    error1 = "Error in <TSystem::ExpandFileName>: input: $HOME/.root.mimes, output: $HOME/.root.mimes"
                    error2 = "Error in <TNetXNGFile::Open>: [ERROR] Server responded with an error: [3006] tried hosts option not supported."
                    if (line[:-1] != error1) and (line[:-1] != error2):
                        count_error += line.count("Error in")
                        count_error += line.count("Traceback")
            if count_error != 0:
                print('Errors in job ' + str(idxJob) + ': ' + str(count_error))
        else:
            print('Errors in job ' + str(idxJob) + ': no file')

    print("\n")
"""