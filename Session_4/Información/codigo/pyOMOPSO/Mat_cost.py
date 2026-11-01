import numpy as np

def mat_cost(pop):

    Costs = []

    for i in range(0, len(pop)):
        Costs.append(pop[i]['cost'].T)

    c_mat = np.array(list(Costs))

    return c_mat


    




   