An example of synthesis tool from Aiger http://fmv.jku.at/aiger/ circuits format.

Installation requirements:
  - pycudd library: http://bears.ece.ucsb.edu/pycudd.html
  - swig library: http://www.swig.org/
  - (probably) python2.7 headers

After installing pycudd library add cudd libraries into your LD_LIBRARY_PATH:

export LD_LIBRARY_PATH=/path/to/pycudd2.0.2/cudd-2.4.2/lib

To setup AIGER parser, run aiger_swig/make_swig.sh

To run:

./aisy.py -h

Some self-testing functionality is included in ``run_status_tests.py``.

Email me in case of questions/suggestions/bugs: ayrat.khalimovatgmail.
