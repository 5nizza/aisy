#!/bin/bash

synt_2_hwmcc=/home/ayrat/projects/spec-framework/synt_2_hwmcc.py
fairness_2_justice=/home/ayrat/projects/spec-framework/fairness_2_justice.py
MC=/home/ayrat/projects/iimc-2.0/iimc

tmp_file=`mktemp --suffix .aag`
$synt_2_hwmcc $1 | $fairness_2_justice > $tmp_file
echo "conversion succeeded: " $1 " converted to " $tmp_file
echo "model checking:"
$MC $tmp_file
