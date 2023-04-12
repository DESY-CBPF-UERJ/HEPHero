#!/usr/bin/env python

# A very simple script to divide an input file into individual pieces so you can run brilcalc over each piece
# in parallel rather than having one huge job.

import sys, json, argparse

parser = argparse.ArgumentParser(description="Split an input JSON file into separate pieces.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("inputFile", help="Input JSON file")
parser.add_argument("-o", "--output-file", help="Output file name template", default="split_file_%d.json")
parser.add_argument("-r", "--runs-per-file", help="Number of runs in each output file", type=int, default=10)
args = parser.parse_args()

infile = args.inputFile
runs_per_file = args.runs_per_file
output_template = args.output_file

if output_template.find("%d") < 0:
    print "Error: output file name template should include %d (example: split_file_%d.json)"
    sys.exit(1)

# First, read in the input JSON file.
with open(infile) as json_input:
    parsedJSON = json.load(json_input)

# Now segment it.
input_runs = sorted(parsedJSON.keys())
counter = 1
for i in range(0, len(input_runs), runs_per_file):
    these_runs = input_runs[i:i+runs_per_file]
    outfile_name = output_template % (counter)
    outfile = open(outfile_name, "w")
    outfile.write("{"+",\n".join("\""+x+"\": "+json.dumps(parsedJSON[x]) for x in these_runs)+"}\n")
    outfile.close()
    counter += 1

print len(input_runs),"runs processed"
