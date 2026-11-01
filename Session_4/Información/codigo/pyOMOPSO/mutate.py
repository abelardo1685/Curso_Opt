import numpy as np
from copy import deepcopy
import copy


def mutate(x, pm, VarMin, VarMax, i, mutation):

    # free election
    r = np.random.randint(1,4)
   
    #encuentra cuantas mutaciones existen ya de tal metodo
    cant_actual = np.sum((r == mutation)*1)

    if cant_actual < len(mutation)/3:
        mutation[i,0] = r
    else:
        # Repite de nuevo el metodo a elegir
        while cant_actual >= len(mutation)/3:

            # vuelve a escribir los metodos de mutacion disponibles
            vector = [1, 2, 3];  
            vector[r-1] = 0
            vector = np.array(vector)
            new_vector = vector[vector!=0]

            if np.random.uniform(0,1) < 0.5:
                r = copy.copy(new_vector[0])
            else:
                r = copy.copy(new_vector[1])

            cant_actual = np.sum((r == mutation)*1)
            mutation[i,0] = r

    if r == 1:
        mutation_rate = pm
    elif r == 2:
        mutation_rate = 1

    # Generar mutacion

    if r == 1 or r == 2:

         #x: vector de variables de decision por cada particula
         # NUmero de variables
        nVar=len(x);

        # Numero aleatorio entero
        j= np.random.randint(0,nVar)

        # Mutacion respecto al rango maximo
        dx = mutation_rate*(VarMax-VarMin)

        # Selecciona el limite inferior de una variable de decision escogida al azar.
        lb=x[j]-dx

        ## Modificar valores de frontera de acuerdo a su existencia.

        # en caso de ser un valor menor al minimo este se modifica para que
        # solo toque la frontera inferior, impidiendo que la particula salga
        # del lindero de variables posibles.

        if lb<VarMin:
            lb=VarMin

        # no permitir que salga de la frontera maxima
        ub=x[j]+dx

        if ub > VarMax:
            ub=VarMax

        ## Se modifica la variable de decision escogida aleatoriamente.
        xnew = x.copy()
        xnew[j]=np.random.uniform(lb,ub)
        NewSol = []
       # NewSol = [] deepcopy(pop[i])
        #NewSol= {'position':xnew}
        NewSol= deepcopy({'position':xnew})
        #NewSol_pos = xnew
    elif r == 3: # no mutation
        xnew=x.copy()
        #NewSol_pos = xnew
        NewSol= []
        NewSol= deepcopy({'position':xnew})
        #NewSol_pos = xnew
   
    return NewSol, mutation