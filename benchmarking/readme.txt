This is a benchmark framework which might be useful for evaluating
synthesis tools developed for the lecture
  AK Design and Verification 2013
at the
  Institute for Applied Information Processing and Communications,
  Graz University of Technology.

Version: 1.0.1
Created by Robert Koenighofer, robert.koenighofer@iaik.tugraz.at

It requires Linux/Unix or Windows with Cygwin and the tools
 - GNU time and
 - ulimit (part of bash)
 - build-essentials (gcc, make, ...)

To run the benchmarks on your tool you have to:
 - compile the model-checker by opening a shell in the directory
   'ext_tools/blimc' and executing the commands:
   > ./configure
   > make
   This should create the executable 'ext_tools/blimc/blimc'
 - Change the variable COMMAND in call_synth_tool.sh to some command
   which invokes your tool.
 - Maybe modify the variable GNU_TIME in performance_test.sh to
   point to the GNU time tool.
 - Call performance_test.sh from a bash.
 - A directory 'results' will be created. It contains:
   - results_<timestamp>.txt: contains the execution time of your tool
     on the different benchmarks
   - results_<timestamp>: is a directory which will contain the
     synthesis results (the verilog implementations) of your tool.

Optionally, you can:
 - Change the timeout by changing TIME_LIMIT in performance_test.sh.
 - Change the memory limit by changing MEMORY_LIMIT in performance_test.sh.
 - Add additional benchmarks by adding them to the list FILES in
   performance_test.sh.
Have a look at the comments in the scripts for additional help.
   
You can easily copy results of a run into an EXCEL table as follows:
 - In the directory 'results', execute:
   > python ./log_to_table.py ./results_<timestamp>.txt ./results_<timestamp>.csv
 - Open the table results/results.ods/xls/xlsx
 - Replace the yellow area with the data from results_<timestamp>.csv
Having the data in an EXCEL table allows you to compute metrics or charts
over your data easily.


Any questions? Help Required?
Send me an email: robert.koenighofer@iaik.tugraz.at
 
