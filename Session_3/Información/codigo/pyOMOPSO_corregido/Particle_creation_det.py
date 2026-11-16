# v2 (Curso OMOPSO 2.0, fase F0): version serial — se elimino la dependencia
# de Ray: lanzaba UNA sola tarea remota con todo el bucle dentro, por lo que
# solo agregaba overhead. Serial ademas hace reproducible la corrida con seed.

import numpy as np

def particle_creation(problem, empty_particle, OMOPSO):
    
    nPop = OMOPSO['nPop']

    # Define the function. Each remote function will be executed 
    # in a separate process.
    def particle_create(problem, empty_particle):

        VarMin = problem['VarMin']
        VarMax = problem['VarMax']
        nVar   = problem['nVar']
        CostFunction = problem['CostFunction']
        #K_alpha = problem['alpha']
        #K_beta = problem['beta']
        
                
        pop = empty_particle.copy()

        pop['position'] = np.random.uniform(VarMin, VarMax, nVar)
        pop['velocity'] = np.zeros(nVar)
        pop['cost'] = CostFunction(pop['position'])
        pop['best_position'] = pop['position'].copy()
        pop['best_cost'] = pop['cost']

        return pop

    return [particle_create(problem, empty_particle) for _ in range(nPop)]