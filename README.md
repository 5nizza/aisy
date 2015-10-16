## What?

Synthesis tool using a modified [AIGER](http://fmv.jku.at/aiger/) format as input.

The input format is described in the paper published at SYNT.


## Setup
Prerequisites:

  - pycudd library: http://bears.ece.ucsb.edu/pycudd.html
    (tested with version 2.0.2)
  - swig library: http://www.swig.org/
    (tested with versions 2.0.11)
  - (probably) python2.7 headers
  - checked on Ubuntu 12.04 and 14.04, likely works with others
  - testing script `run_tests.py` requires `spec-framework` from https://bitbucket.org/art_haali/spec-framework

After installing pycudd library add cudd libraries into your `LD_LIBRARY_PATH`:

    export LD_LIBRARY_PATH=/path/to/pycudd2.0.2/cudd-2.4.2/lib

which is automated in `setup.sh` that is run with `. ./setup.sh`.

Then compile AIGER parser, by running

    aiger_swig/make_swig.sh


## Run

    ./aisy.py -h

If you get 

`ImportError: libcudd.so: cannot open shared object file: No such file or directory`

go to step `Setup`.


## Test
To run tests without model checking the models synthesized:

    ./run_tests.py

To run with following model checking:

    ./run_tests.py --mc

Do not forget to set up paths in `run_tests.py` first.


## Contact
Email to ayrat.khalimovatgmail or post messages here.


## Authors
Ayrat Khalimov, and many thanks to Robert Koenighofer for fruitful discussions.

