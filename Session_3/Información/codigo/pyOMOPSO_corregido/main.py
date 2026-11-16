"""

Copyright (c) 2022, Abelardo Rodríguez Pretelín (www.IA.com)
All rights reserved. Please read the "license.txt" for usage terms.
__________________________________________________________________________

Project Code: parOMOPSO
Project Title: Implementation of a parallel version of the Optimal multi-objective particle swarm optimizer in Python. 
According to [2], OMOPSO is one of the top-performing PSO algorithms.
References:

[1] Sierra, M. R. and C. A. Coello Coello (2005). Improving PSO-based Multi-Objective Optimization using Crowding, Mutation and ε-Dominance. Evolutionary Multi-Criterion Optimization, pp. 505-519.
[2] Durillo, J. J., J. Garcia-Nieto, A. J. Nebro, C. A. Coello Coello, F. Luna, and E. Alba (2009). Multi-Objective Particle Swarm Optimizers: An Experimental Comparison. Evolutionary Multi-Criterion Optimization, pp. 495-509.

Developer: A. Rodríguez-Pretelín (Head of IA Team)

Contact Info: abelardo.IA@gmail.com """

# 1.- Define Optimization Problem. A Sample Cost Function
# The cost function must be declared either inside the dtlz3 class function or as individual function

Prueba = dtlz3()
Test_function = Prueba.Test_function

# Define Optimization Problem

problem = {
        'CostFunction': Test_function,
        'nVar': 3,
        'VarMin': 0,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VarMax': 1,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
    };

# Define algorithm variables
OMOPSO = {
        'MaxIter': 150,
        'nPop': 40,
        'w1': 0.1,
        'w2': 0.5,
        'e': 0.001,
        'c1': 1.5,
        'c2': 2,
        'nGrid': 7,
        'alpha': 0.1,
        'wdamp': 0.99,
        'r1': 0,
        'r2': 1,
    }
    
# Running par_OMOPSO
yp.tic()
print('Running parOMOPSO ...')
e_file, rep = parOMOPSO(problem, OMOPSO);
print()
yp.toc()
print()

# Final Result




