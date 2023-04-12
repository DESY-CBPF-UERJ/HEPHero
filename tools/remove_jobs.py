import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--selection")
parser.add_argument( 
  "-l", "--dirlist",  # name on the argument
  nargs="*"  # 0 or more values expected => creates a list
)
args = parser.parse_args()


with open('analysis.txt') as f:
    analysis = f.readline()
    
outpath = os.environ.get("HEP_OUTPATH")
basedir = os.path.join(outpath, analysis, args.selection)

for folder in args.dirlist:
    removeCommand = "rm -rf " + os.path.join(basedir, folder)
    print(removeCommand)
    os.system(removeCommand)
