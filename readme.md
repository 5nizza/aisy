Simple synthesis tool from Aiger http://fmv.jku.at/aiger/ circuits format.      
Some description of the tool is available at        
https://verify.iaik.tugraz.at/research/bin/view/Ausgewaehltekapitel/AkdvExercise14           
Some slides        
https://verify.iaik.tugraz.at/research/pub/Ausgewaehltekapitel/AkdvExercise14/aisy.pdf        
and even a recorded video of presentation given in 2014 in our TU Graz class           
https://bigfiles.iaik.tugraz.at/get/9c3810d6366b34e721d912b3489c2798         

## Setup ##
Prerequisites:

  - pycudd library: http://bears.ece.ucsb.edu/pycudd.html
    (tested with version 2.0.2)
  - swig library: http://www.swig.org/
    (tested with versions 2.0.9, and seems to work with 3.0.2)
  - (probably) python2.7 headers
  - checked on Ubuntu 12.04, likely works with others

After installing pycudd library add cudd libraries into your LD_LIBRARY_PATH:     
`export LD_LIBRARY_PATH=/path/to/pycudd2.0.2/cudd-2.4.2/lib`

Then compile AIGER parser, by running        
`aiger_swig/make_swig.sh`

## Run ##
`./aisy.py -h`

Some self-testing functionality is included in `run_status_tests.py`.

## Contact ##
Email to ayrat.khalimovatgmail or post messages here.

## Authors ##
Ayrat Khalimov, Robert Koenighoffer, and SCOS group at TU Graz.
