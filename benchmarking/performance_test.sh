#!/bin/bash

# This is a benchmark framework which might be useful for evaluating
# synthesis tools developed for the lecture
#   AK Design and Verification 2013
# at the
#   Institute for Applied Information Processing and Communications,
#   Graz University of Technology.
#
# Version: 1.0.2
# Created by Robert Koenighofer, robert.koenighofer@iaik.tugraz.at 
# Comments/edits by Swen Jacobs, swen.jacobs@iaik.tugraz.at

# This directory:
DIR=`dirname $0`/

# Time limit in seconds:
TIME_LIMIT=600
# Memory limit in kB:
MEMORY_LIMIT=2000000

# Maybe change the following line to point to GNU time:
GNU_TIME="/usr/bin/time"
MODEL_CHECKER="$DIR/ext_tools/blimc/blimc"          
MODEL_CHECKER_OPTIONS="15" # For blimc, this is the bound for BMC
SYNT_CHECKER="$DIR/ext_tools/syntactic_checker.py"

# The directory where the benchmarks are located:
BM_DIR="${DIR}benchmarks/"

REAL=10
UNREAL=20

# The benchmarks to be used.
# The files have to be located in ${BM_DIR}.
FILES=(
  unr                       $UNREAL
  eq1                       $REAL
  ex1                       $REAL
  ex2                       $REAL
  add2                      $REAL
  add2_o                    $REAL
#  add4                      $REAL
#  add4_o                    $REAL
  add6                      $REAL
  add6_o                    $REAL
#  add8                      $REAL
#  add8_o                    $REAL
  add10                     $REAL
  add10_o                   $REAL
  add12                     $REAL
  add12_o                   $REAL
  add14                     $REAL
  add14_o                   $REAL
  add16                     $REAL
  add16_o                   $REAL
  add18                     $REAL
  add18_o                   $REAL
  add20                     $REAL
  add20_o                   $REAL
  cnt2_u                    $UNREAL
  cnt2_u_o                  $UNREAL
  cnt2m                     $REAL
  cnt2m_o                   $REAL
#  cnt3m                     $REAL
#  cnt3m_o                   $REAL
#  cnt4m                     $REAL
#  cnt4m_o                   $REAL
  cnt5m                     $REAL
  cnt5m_o                   $REAL
#  cnt6m                     $REAL
#  cnt6m_o                   $REAL
#  cnt7m                     $REAL
#  cnt7m_o                   $REAL
#  cnt8m                     $REAL
#  cnt8m_o                   $REAL
#  cnt9m                     $REAL
#  cnt9m_o                   $REAL
  cnt10m                    $REAL
  cnt10m_o                  $REAL
#  cnt11m                    $REAL
#  cnt11m_o                  $REAL
  cnt15m                    $REAL
  cnt15m_o                  $REAL
  cnt20m                    $REAL
  cnt20m_o                  $REAL
#  cnt25m                    $REAL
  cnt25m_o                  $REAL
#  cnt30m                    $REAL
  cnt30m_o                  $REAL
  mv2                       $REAL
  mv2_o                     $REAL
  mv2s                      $REAL
  mv2s_o                    $REAL
  mv4                       $REAL
  mv4_o                     $REAL
  mv4s                      $REAL
  mv4s_o                    $REAL
  mv8                       $REAL
  mv8_o                     $REAL
  mv8s                      $REAL
  mv8s_o                    $REAL
  mv12                      $REAL
  mv12_o                    $REAL
  mv12s                     $REAL
  mv12s_o                   $REAL
#  mv16                      $REAL
  mv16_o                    $REAL
  mv16s                     $REAL
  mv16s_o                   $REAL
  mv18s                     $REAL
  mv18s_o                   $REAL
#  mv20                      $REAL
  mv20_o                    $REAL
  mv20s                     $REAL
  mv20s_o                   $REAL
  mv22s                     $REAL
  mv22s_o                   $REAL
  mv24s                     $REAL
  mv24s_o                   $REAL
  mv28s                     $REAL
  mv28s_o                   $REAL
  comb_mult2                $REAL
  comb_mult4                $REAL
  comb_mult8                $REAL
  comb_mult16               $REAL
  amba02_new_08n_unreal     $UNREAL
  amba02_new_08n_unreal_o   $UNREAL
  amba02_new_09n            $REAL
  amba02_new_09n_o          $REAL
  amba03_new_08n_unreal     $UNREAL
  amba03_new_08n_unreal_o   $UNREAL
  amba03_new_09n            $REAL
  amba03_new_09n_o          $REAL
#  amba04_new_24n_unreal     $UNREAL
#  amba04_new_24n_unreal_o   $UNREAL
  amba04_new_25n            $REAL
  amba04_new_25n_o          $REAL
#  amba05_new_16n_unreal     $UNREAL
#  amba05_new_16n_unreal_o   $UNREAL
#  amba05_new_17n            $REAL
  amba05_new_17n_o          $REAL
#  amba06_new_20n_unreal     $UNREAL
#  amba06_new_20n_unreal_o   $UNREAL
#  amba06_new_21n            $REAL
  amba06_new_21n_o          $REAL
#  amba07_new_24n_unreal     $UNREAL
#  amba07_new_24n_unreal_o   $UNREAL
#  amba07_new_25n            $REAL
#  amba07_new_25n_o          $REAL
  genbuf01_03_unreal        $UNREAL
  genbuf01_03_unreal_o      $UNREAL
  genbuf01_04               $REAL
  genbuf01_04_o             $REAL
  genbuf02_03_unreal        $UNREAL
  genbuf02_03_unreal_o      $UNREAL
  genbuf02_04               $REAL
  genbuf02_04_o             $REAL
#  genbuf03_03_unreal        $UNREAL
#  genbuf03_03_unreal_o      $UNREAL
  genbuf03_04               $REAL
  genbuf03_04_o             $REAL
#  genbuf04_03_unreal        $UNREAL
#  genbuf04_03_unreal_o      $UNREAL
  genbuf04_04               $REAL
  genbuf04_04_o             $REAL
#  genbuf05_04_unreal        $UNREAL
#  genbuf05_04_unreal_o      $UNREAL
#  genbuf05_05               $REAL
  genbuf05_05_o             $REAL
#  genbuf06_05_unreal        $UNREAL
#  genbuf06_05_unreal_o      $UNREAL
#  genbuf06_06               $REAL
  genbuf06_06_o             $REAL
#  genbuf07_06_unreal        $UNREAL
#  genbuf07_06_unreal_o      $UNREAL
#  genbuf07_07               $REAL
  genbuf07_07_o             $REAL
#  genbuf08_07_unreal        $UNREAL
#  genbuf08_07_unreal_o      $UNREAL
#  genbuf08_08               $REAL
  genbuf08_08_o             $REAL
#  genbuf09_08_unreal        $UNREAL
#  genbuf09_08_unreal_o      $UNREAL
#  genbuf09_09               $REAL
  genbuf09_09_o             $REAL
#  genbuf10_09_unreal        $UNREAL
#  genbuf10_09_unreal_o      $UNREAL
#  genbuf10_10               $REAL
#  genbuf10_10_o             $REAL
)

CALL_SYNTH_TOOL=${DIR}call_synth_tool.sh
TIMESTAMP=`date +%s`
RES_TXT_FILE="${DIR}results/results_${TIMESTAMP}.txt"
RES_DIR="${DIR}results/results_${TIMESTAMP}/"
mkdir -p "${DIR}results/"
mkdir -p ${RES_DIR}

ulimit -m ${MEMORY_LIMIT} -v ${MEMORY_LIMIT} -t ${TIME_LIMIT}
for element in $(seq 0 2 $((${#FILES[@]} - 1)))
do
     file_name=${FILES[$element]}
     infile_path=${BM_DIR}${file_name}.aag
     outfile_path=${RES_DIR}${file_name}_synth.aag
     correct_real=${FILES[$element+1]}
     echo "Synthesizing ${file_name}.aag ..."
     echo "=====================  $file_name.aag =====================" 1>> $RES_TXT_FILE

     #------------------------------------------------------------------------------
     # BEGIN execution of synthesis tool
     echo " Running the synthesizer ... "
     ${GNU_TIME} --quiet --output=${RES_TXT_FILE} -a -f "Synthesis time: %e sec (Real time) / %U sec (User CPU time)" ${CALL_SYNTH_TOOL} $infile_path $outfile_path
     exit_code=$?
     echo "  Done running the synthesizer. "
     # END execution of synthesis tool

     if [[ $exit_code == 137 ]];
     then
         echo "  Timeout!"
         echo "Timeout: 1" 1>> $RES_TXT_FILE
         continue
     else
         echo "Timeout: 0" 1>> $RES_TXT_FILE
     fi

     if [[ $exit_code != $REAL && $exit_code != $UNREAL ]];
     then
         echo "  Strange exit code: $exit_code (crash or out-of-memory)!"
         echo "Crash or out-of-mem: 1 (Exit code: $exit_code)" 1>> $RES_TXT_FILE
         continue
     else
         echo "Crash or out-of-mem: 0" 1>> $RES_TXT_FILE
     fi

     #------------------------------------------------------------------------------
     # BEGIN analyze realizability verdict
     if [[ $exit_code == $REAL && $correct_real == $UNREAL ]];
     then
         echo "  ERROR: Tool reported 'realizable' for an unrealizable spec!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
         echo "Realizability correct: 0 (tool reported 'realizable' instead of 'unrealizable')" 1>> $RES_TXT_FILE
         continue
     fi
     if [[ $exit_code == $UNREAL && $correct_real == $REAL ]];
     then
         echo "  ERROR: Tool reported 'unrealizable' for a realizable spec!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
         echo "Realizability correct: 0 (tool reported 'unrealizable' instead of 'realizable')" 1>> $RES_TXT_FILE
         continue
     fi
     if [[ $exit_code == $UNREAL ]];
     then
         echo "  The spec has been correctly identified as 'unrealizable'."
         echo "Realizability correct: 1 (unrealizable)" 1>> $RES_TXT_FILE
     else
         echo "  The spec has been correctly identified as 'realizable'."
         echo "Realizability correct: 1 (realizable)" 1>> $RES_TXT_FILE

         # END analyze realizability verdict

         #------------------------------------------------------------------------------
         # BEGIN syntactic check
         echo " Checking the synthesis result syntactically ... "
         if [ -f $outfile_path ];
         then
             echo "  Output file has been created."
             python $SYNT_CHECKER $infile_path $outfile_path
             exit_code=$?
             if [[ $exit_code == 0 ]];
             then
               echo "  Output file is OK syntactically."
               echo "Output file OK: 1" 1>> $RES_TXT_FILE
             else
               echo "  Output file is NOT OK syntactically."
               echo "Output file OK: 0" 1>> $RES_TXT_FILE
             fi
         else
             echo "  Output file has NOT been created!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
             echo "Output file OK: 0 (no output file created)" 1>> $RES_TXT_FILE
             continue
         fi
         # TODO: perform syntactic check here.
         # END syntactic check

         #------------------------------------------------------------------------------
         # BEGIN model checking
         echo -n " Model checking the synthesis result ... "
         ${GNU_TIME} --quiet --output=${RES_TXT_FILE} -a -f "Model-checking time: %e sec (Real time) / %U sec (User CPU time)" $MODEL_CHECKER $MODEL_CHECKER_OPTIONS $outfile_path > /dev/null 2>&1
         check_res=$?
         echo " done. "
         if [[ $check_res == 20 ]];
         then
             echo "  Model-checking was successful."
             echo "Model-checking: 1" 1>> $RES_TXT_FILE
         else
             echo "  Model-checking the resulting circuit failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
             echo "Model-checking: 0 (exit code: $check_res)" 1>> $RES_TXT_FILE
         fi
         # END end checking

         #------------------------------------------------------------------------------
         # BEGIN determining circuit size
         aig_header_in=$(head -n 1 $infile_path)
         aig_header_out=$(head -n 1 $outfile_path)
         echo "Raw AIGER input size: $aig_header_in" 1>> $RES_TXT_FILE
         echo "Raw AIGER output size: $aig_header_out" 1>> $RES_TXT_FILE
         # END determining circuit size           
     fi
done
