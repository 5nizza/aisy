#!/bin/bash

# build aiger.so
gcc -c -fpic ./aiger.c && gcc -shared -o aiger.so aiger.o


#build swig wrappers
interface_file=aiger.i

swig -python $interface_file && 
gcc -O2 -fPIC -c aiger.c && 
gcc -O2 -fPIC -c aiger_wrap.c -I/usr/include/python2.7/ && 
gcc -shared aiger.o aiger_wrap.o -o _aiger_wrap.so 
