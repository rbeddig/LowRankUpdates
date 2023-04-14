# C++ Prototype for preconditioner low-rank updates for a Boussinesq system

The current prototype is a buoyancy Boussinesq system, i.e., an incompressible Navier-Stokes system with Coriolis force that is driven through density variations caused by temperature changes.

This code is meant to be used for reproducing the results shown in [1].
The goal of this code is to test different block preconditioners with low-rank updates for the Schur complement preconditioner.


# Installation
First call "cmake .", followed by "make release" (will be compiled in optimized mode) or "make debug" (will be compiled in debug mode).
Then, call "make" to compile the program.

If cmake does not find deal.II, the deal.II-install-path needs to be adapted in line 18 of the file "CMakeLists.txt".

You find the settings for installing the required packages (Trilinos, p4est) in the directory install_files.


# Run the program
Call it with "mpirun -np 1 source/BoussinesqPlanet -p data/params.prm"
"params.prm" needs to be replaced with the name of a parameter file. 

To reproduce the numerical results shown in [1], you can run the following commands:

"./run_table1"

"./run_table2"

"./run_table3"

The results are printed to "numerical_results.tex".


# Dependencies
This code is used with the deal.II-version 9.3.3.
deal.II needs to be compiled with MPI, Trilinos (version 12.14.1) and p4est.
Furthermore, we need the package numdiff (from https://www.nongnu.org/numdiff/).

In the directory "build-files" are configuration files for installing Trilinos, p4est and deal.II. Here, paths to required packages need to be adapted.
The library deal.II should be installed as the last package.


# Authors
The Boussinesq model is implemented by Konrad Simon.
The preconditioners are implemented by Rebekka Beddig.


# References
[1] R.S. Beddig, J. Behrens, S. Le Borne, K. Simon. An error-based low-rank correction for pressure Schur complement preconditioners. Proceedings of the YRM & CSE Workshop on Modeling, Simulation & Optimization of Fluid Dynamic Applications, March 21-24, 2022.