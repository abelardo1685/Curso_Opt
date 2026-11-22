import numpy as np
from Dominate_rep import DominateRepository

def DetermineDominationRepository(e_file,e):

    repository = e_file # repositorio actual
    nPop = len(repository)
    
    for i in range(0, nPop):
        repository[i]['is_dominated'] = 0
        
    # Comparar entre el i actual respecto a los siguientes valores 
    # e.g.: valor 1 compararlo con i 2 en adelante
    
    # Dominates: tanto que sea el mayor o menor de entre los dos.
    
    for i in range (0, nPop-1):
        for j in range(i+1, nPop):
        
            # Caso       if  (a  dominates  b) en lo global como en lo particular
            if DominateRepository(repository[i], repository[j], e):
                repository[j]['is_dominated'] = 1


            # Caso         (b   dominates  a) en lo global como en lo particular
            # V3 (2026-10-16): «elif» en lugar de «if». Dos soluciones con la misma caja ε se «dominaban» mutuamente y
            # ambas se borraban; con copias repetidas de la única solución factible el archivo quedaba vacío.
            elif DominateRepository(repository[j], repository[i], e):
                repository[i]['is_dominated'] = 1 
        
        
    # Erasing dominated particles
    repository = [sub for sub in repository if sub['is_dominated'] == 0]
    
    return repository