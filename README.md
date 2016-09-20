## What?

Synthesis tool using a modified [AIGER][AIGER] based format as input.
The format allows for 1-streett specifications.
Described [here][spec-framework], and is derived from 
[SYNTCOMP][SYNTCOMP] format.


## Prerequisites

  - pycudd library: http://bears.ece.ucsb.edu/pycudd.html
    (tested with version 2.0.2)
  - swig library: http://www.swig.org/
    (tested with versions 2.0.11)
  - (probably) python2.7 headers
  - checked on Ubuntu 14.04 and 16.04
  - testing script `run_func_tests.py` requires 
    [`spec-framework`][spec-framework] 
    and [IIMC][IIMC] if you want to model check the results

## Setup

1.
Run

    ./configure.py

It will create config files `config.py` and `setup.sh` 
-- update them according to your setup.

2.
File `setup.sh` puts cudd library into your `LD_LIBRARY_PATH`:

    export LD_LIBRARY_PATH=/path/to/pycudd2.0.2/cudd-2.4.2/lib

So, run it (`. ./setup.sh`) before using `aisy.py`.

3.
Then compile AIGER parser, by running

    aiger_swig/make_swig.sh


## Run

    ./aisy.py -h

If you get 

`ImportError: libcudd.so: cannot open shared object file: No such file or directory`

then run `. ./setup.sh`.


## Test
To run tests without model checking of the models synthesized:

    ./run_func_tests.py

To run it with model checking:

    ./run_func_tests.py --mc

(from the tool root directory)

## Contact
Gmail me: ayrat.khalimov


## Authors
Ayrat Khalimov.
Many thanks
to Kurt Nistelberger for implementation of optimizations "vector compose", "variable elimination", and "caching of ANDs";
to Robert Koenighofer for fruitful discussions.



[IIMC]: http://ecee.colorado.edu/wpmu/iimc/
[spec-framework]: https://github.com/5nizza/spec-framework
[AIGER]: http://ecee.colorado.edu/wpmu/iimc/
[SYNTCOMP]: http://arxiv.org/abs/1405.5793
