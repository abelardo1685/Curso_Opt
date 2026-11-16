import numpy as np
from Dominate import Dominate

def determine_domination(pop):
    
    
    Pop_Size = len(pop)

    for i in range(0, Pop_Size):

        pop[i]['is_dominated'] = 0


    for i in range(0, Pop_Size-1):
        for j in range(i+1, Pop_Size):

            if Dominate(pop[i], pop[j]):
                pop[j]['is_dominated'] = 1
                    
            if Dominate(pop[j], pop[i]):
                pop[i]['is_dominated'] = 1
            
    return pop
