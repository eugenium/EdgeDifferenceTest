# EdgeDifferenceTest
Testing for Differences. This has code and routines associated with the paper https://arxiv.org/pdf/1512.08643.pdf

### Dependencies
This code is written in python with wrappers on cvxopt,mosek, and R packages.

The following packages and their python bindings are needed: cvxopt and mosek , rpy2
R should be installed as well as the package genlasso which is used
The current project was developed using versions cvxopt 1.1.4 and mosek 7
Recent version of these packages (can be installed with acanaconda more easily) but have not thoroughly tested yet

### Run
Difference_Test_Sim.py to get started

Note:  Current code has not been optimized and requires further packaging and documentation. mosek produces blank output that takes over the command line thus intermediate printing is best done to a file, this may be resolved in the future by switching and optimizing the QP solver 


email: eugene.belilovsky@inria.fr 
