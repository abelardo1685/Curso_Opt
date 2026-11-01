import numpy as np

def mat_grid_index(pop):

    grid_index = []
    
    for i in range(0,len(pop)):
        grid_index.append(pop[i]['grid_index'])     

    return grid_index
