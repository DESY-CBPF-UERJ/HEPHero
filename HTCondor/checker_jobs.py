import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cluster-id", type=str, help="Cluster ID where condor is running your jobs.")
args = parser.parse_args()

if args.cluster_id is None:
    quit("Run -h option and see how to use -c argument.")
else:
    cluster_id = args.cluster_id

cwd = os.getcwd()
log_file = os.path.join(cwd, "run_" + cluster_id + ".log")

count_submitted = 0
count_terminated = 0
count_aborted = 0

with open(log_file) as f:
    for line in f:
        count_submitted += line.count("submitted")
        count_terminated += line.count("terminated.")
        count_aborted += line.count("aborted")
        
print('\nJobs submitted:  ' + str(count_submitted))
print('Jobs terminated: ' + str(count_terminated))
print('Jobs aborted:    ' + str(count_aborted) + '\n')

for idxJob in range(count_submitted):
    err_filename = "jobs_log/run_" + str(cluster_id) + "_" + str(idxJob) + ".err"
    err_file = os.path.join(cwd, err_filename)
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
