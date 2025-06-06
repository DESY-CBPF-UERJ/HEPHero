import os
import sys
import argparse

#======SETUP=======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--did")
parser.set_defaults(did=None)
parser.add_argument("-c", "--container")
parser.set_defaults(container=None)
parser.add_argument("-t", "--T2")
parser.set_defaults(T2="T2_BR_UERJ")
parser.add_argument("--rules", dest='rules_flag', action='store_true')
parser.set_defaults(rules_flag=False)
args = parser.parse_args()

user = os.environ.get("STORAGE_USER")

if args.rules_flag:
    os.system("rucio list-rules --account " + user)
    sys.exit()

if args.container is not None:
    os.system("rucio add-container user."+user+":/Analysis/Datasets"+args.container+"/USER")

total_size = 0
with open(args.did) as f:
    for line in f:
        print(" ")
        print(line[:-1])
        os.system("rucio stat " + line[:-1] + " > did_stat.txt")
        with open("did_stat.txt") as f_stat:
            for stat_line in f_stat:
                if stat_line[0:5] == "bytes":
                    if args.container is not None:
                        os.system("rucio attach user."+user+":/Analysis/Datasets"+args.container+"/USER " + line[:-1])
                    did_size = int(stat_line.split(": ")[1])/1000000000
                    total_size += did_size
                    print("did size = ", did_size, " GB")
os.system("rm did_stat.txt")
print(" ")
print("total size = ", total_size/1000, " TB")

if args.container is not None:
    os.system("rucio add-rule user."+user+":/Analysis/Datasets"+args.container+"/USER 1 " + args.T2)

print(" ")
os.system("rucio list-account-usage " + user)



