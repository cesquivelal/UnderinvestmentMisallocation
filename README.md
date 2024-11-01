# UnderinvestmentMisallocation
Replication code for "Underinvestment and Misallocation Under Sovereign Risk" by Carlos Esquivel

January, 2024:

https://cesquivelal.github.io/Esquivel_UninvMiss.pdf

The code is written in the Julia language, version 1.7.2 and uses the following packages:
      Distributed, Parameters, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, QuadGK, Sobol, Roots, NLsolve, Plots

The file Primitives.jl defines all objects and functions that are used to solve and simulate the model.

The folders Planner and Decentralized contain data with the solution to the planner's problem and decentralized equilibrium, respectively. These solutions are for the calibration specified in the paper. The file mainSolve.jl in each folder generates these solutions using the other auxiliary files in each folder.

The file PlotsCrisis.jl contains additional functions to generate simulated paths for impulse-response functions and European debt crisis using the model solution.

The file FiguresAndTables.jl creates all the figures in the paper and computes the values reported in simulation tables using the data with solutions in folders Planner and Decentralized.
