# Interactions between MDF and ACT-R

This directory contains examples of ACT-R models converted to MDF. The ACT-R
models [count.lisp](count.lisp) and [addition.lisp](addition.lisp) are based on 
the [ACT-R tutorial](http://act-r.psy.cmu.edu/software/).

The script [run_example.py](run_example.py) can be run using
`python run_examples.py count.lisp`
to create the MDF .json and .yaml files for the given example and execute it
using the MDF scheduler. The current implementation uses the 
[ccmsuite](https://github.com/tcstewar/ccmsuite) library, which must be installed 
for ACT-R examples to run using the scheduler.
