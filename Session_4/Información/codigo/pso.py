# pso.py — Optimización por Enjambre de Partículas (PSO) del curso
#
# Reconstrucción del módulo usado por los notebooks de la Sesión 2 (basado en la
# implementación de Yarpiz, ypea127), extendido igual que el "Paso a Paso":
#   - 'MinProblem': 1 minimiza (por defecto), 0 maximiza
#   - 'ResFunction': función de reparación de variables (p. ej. fijar p3 = 600)
#
# Uso:
#   gbest, pop = pso.PSO(problem, MaxIter=200, PopSize=50,
#                        c1=1.5, c2=2, w=1, wdamp=0.995)
#
# Después de cada corrida, pso.historial guarda el mejor costo de cada iteración
# (sirve para dibujar la curva de convergencia).

import time
import numpy as np

_tic_inicio = None
historial = []          # mejor costo por iteración de la última corrida

def tic():
    global _tic_inicio
    _tic_inicio = time.time()

def toc():
    if _tic_inicio is not None:
        print('Elapsed time: {:.2f} s'.format(time.time() - _tic_inicio))

def PSO(problem, MaxIter=100, PopSize=100, c1=1.5, c2=2.0, w=1.0, wdamp=0.995,
        verbose=True):

    # Template de la particula
    empty_particle = {
        'position': None,
        'velocity': None,
        'cost': None,
        'best_position': None,
        'best_cost': None,
    }

    # Extraer información del problema
    CostFunction = problem['CostFunction']
    VarMin = problem['VarMin']
    VarMax = problem['VarMax']
    nVar = problem['nVar']
    Restriction = problem.get('ResFunction', None)
    Minimization = problem.get('MinProblem', 1)

    # Iniciar la solución global
    if Minimization == 1:
        gbest = {'position': None, 'cost': np.inf}
    else:
        gbest = {'position': None, 'cost': -np.inf}

    # Crear población inicial
    pop = []
    for i in range(0, PopSize):
        pop.append(empty_particle.copy())
        pop[i]['position'] = np.random.uniform(VarMin, VarMax, nVar)

        ## Añadir restricción (reparación)
        if Restriction is not None:
            pop[i]['position'] = Restriction(pop[i]['position'])
        ##

        pop[i]['velocity'] = np.zeros(nVar)
        pop[i]['cost'] = CostFunction(pop[i]['position'])
        pop[i]['best_position'] = pop[i]['position'].copy()
        pop[i]['best_cost'] = pop[i]['cost']

        if Minimization == 1:
            if pop[i]['best_cost'] < gbest['cost']:
                gbest['position'] = pop[i]['best_position'].copy()
                gbest['cost'] = pop[i]['best_cost']
        else:
            if pop[i]['best_cost'] > gbest['cost']:
                gbest['position'] = pop[i]['best_position'].copy()
                gbest['cost'] = pop[i]['best_cost']

    historial.clear()

    # PSO Loop principal
    for it in range(0, MaxIter):
        for i in range(0, PopSize):

            pop[i]['velocity'] = w*pop[i]['velocity'] \
                + c1*np.random.rand(nVar)*(pop[i]['best_position'] - pop[i]['position']) \
                + c2*np.random.rand(nVar)*(gbest['position'] - pop[i]['position'])

            pop[i]['position'] += pop[i]['velocity']
            pop[i]['position'] = np.maximum(pop[i]['position'], VarMin)   # clamping
            pop[i]['position'] = np.minimum(pop[i]['position'], VarMax)

            # restricción (reparación)
            if Restriction is not None:
                pop[i]['position'] = Restriction(pop[i]['position'])

            pop[i]['cost'] = CostFunction(pop[i]['position'])

            if Minimization == 1:
                if pop[i]['cost'] < pop[i]['best_cost']:
                    pop[i]['best_position'] = pop[i]['position'].copy()
                    pop[i]['best_cost'] = pop[i]['cost']

                    if pop[i]['best_cost'] < gbest['cost']:
                        gbest['position'] = pop[i]['best_position'].copy()
                        gbest['cost'] = pop[i]['best_cost']
            else:
                if pop[i]['cost'] > pop[i]['best_cost']:
                    pop[i]['best_position'] = pop[i]['position'].copy()
                    pop[i]['best_cost'] = pop[i]['cost']

                    if pop[i]['best_cost'] > gbest['cost']:
                        gbest['position'] = pop[i]['best_position'].copy()
                        gbest['cost'] = pop[i]['best_cost']

        w *= wdamp
        historial.append(gbest['cost'])
        if verbose:
            print('Iteration {}: Best Cost = {}'.format(it, gbest['cost']))

    return gbest, pop
