import numpy as np
from App_order import appearance_sort

def crowding_leaders (rep, nPop):
    
    Costs = []

    for i in range(0, len(rep)):

        Costs.append(rep[i]['cost'].T)

    c_mat = np.array(list(Costs)).T

    nObj       = len(c_mat)   
    n_Part     = np.shape(c_mat)
    n_Part_rep = n_Part[2]

    d           = np.zeros((n_Part_rep,nObj));
    crow_value  = np.zeros((n_Part_rep,1));

    for j in range(0,nObj):

        mat_slices = c_mat[j,:]
        mat_slice  = mat_slices.flatten()

        so = appearance_sort(list(mat_slice))
        cj = np.sort(mat_slice)

        d[so[0],j] = np.inf 

        for i in range(1, n_Part_rep - 1):

            d[so[i],j] = abs(cj[i+1]-cj[i-1])/abs(cj[0]-cj[-1])

        d[so[-1],j] = np.inf # valor mayor igual a infinito

    for i in range(0,n_Part_rep):
            crow_value[i] = np.sum(d[i,:])

    ids     = np.array(crow_value.T).argsort()[-nPop:][::-1]
    ids_rev = np.fliplr(ids)
        # Retain only the best
          # menor a mayor

    ids     = np.array(crow_value.T).argsort()[-nPop:][::-1]
    ids_rev = np.fliplr(ids)
    index   = np.squeeze(ids_rev)
    index_rej = index[nPop:]

    new_repository = []

    for i in range(0,n_Part_rep):
        if (i in index_rej):
            pass
        else:
            new_repository.append(rep[i].copy())

    rep = new_repository

    return rep
