import numpy as np

def mat_dom(pop):

    dom_ind = []

    for i in range(0,len(pop)):
        dom_ind.append(pop[i]['is_dominated'])
    
  
    return dom_ind
        
    
  