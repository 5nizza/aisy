#!/bin/bash

# This is a benchmark framework which might be useful for evaluating
# synthesis tools developed for the lecture
#   AK Design and Verification 2013
# at the
#   Institute for Applied Information Processing and Communications,
#   Graz University of Technology.
#
# Version: 1.0.0
# Created by Robert Koenighofer, robert.koenighofer@iaik.tugraz.at

DIR=`dirname $0`/

# Change the following line to invoke your solver.
# You can use ${DIR} (which contains the path to the
# parent directory of this script) to specify the path.
# $1 contains the input filename (the name of the AIGER-file).
# $2 contains the output filename (your synthesis result, also in
#    AIGER format).
#COMMAND="echo call_synth_tool.sh called with parameters $1 $2"

#other examples:

COMMAND="/home/art_haali/learning/akdv14/aisy-classroom/aisy.py $1 --out=$2"
# COMMAND="${DIR}../bin/my_synth_tool --in=$1 --out=$2"
# COMMAND="${DIR}../../bin/my_tool --input=$1 --verbose=0 -whatever_option"

# In the end, calling this script with two filenames as parameters
# should make your tool start synthesizing the spec in the first file,
# and write your synthesis result into the second file.

#echo $COMMAND
$COMMAND
