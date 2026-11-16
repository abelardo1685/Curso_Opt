import numpy as np
from App_order import appearance_sort
import math

def SelectLeader(rep,nPop):

    Costs = []

    for i in range(0, len(rep)):

        Costs.append(rep[i]['cost'].T)

    c_mat = np.array(list(Costs)).T

    nObj       = len(c_mat)   
    n_Part     = np.shape(c_mat)
    n_Part_rep = n_Part[2]

    d           = np.zeros((n_Part_rep,nObj))
    crow_value  = np.zeros((n_Part_rep,1))

    for j in range(0,nObj):

        mat_slices = c_mat[j,:]
        mat_slice  = mat_slices.flatten()

        so = appearance_sort(list(mat_slice))
        cj = np.sort(mat_slice)

        d[so[0],j] = np.inf 

        for i in range(1, n_Part_rep - 1):

            d[so[i],j] = abs(cj[i+1]-cj[i-1])/abs(cj[0]-cj[-1])

        d[so[-1],j] = np.inf # valor mayor igual a infinito
        
  
    ## Crow-Comparison-Operator
    for i in range(0,n_Part_rep):
        rep[i]['crowding_distance'] = np.sum(d[i,:])
        crow_value[i] = np.sum(d[i,:])
 

    ids     = np.array(crow_value.T).argsort()[-nPop:][::-1]
    ids_rev = np.fliplr(ids)

    # Sort Based on Crowding Distance

    CDSO = [] # list the crowd distance values to obtain the one with the minimum amount of particles around

    for i in range(0,n_Part_rep):
        index = ids_rev[0,i]

        if math.isinf(rep[index]['crowding_distance']) != 1:
            CDSO.append(rep[index])

    
    # V3 (2026-10-16): con dos o menos soluciones en el repositorio todas tienen hacinamiento infinito y CDSO queda vacía
    # (IndexError). En ese caso el líder se toma al azar del repositorio, como en el MOPSO original.
    leader = CDSO[0] if CDSO else rep[np.random.randint(len(rep))]
    

    return leader, rep