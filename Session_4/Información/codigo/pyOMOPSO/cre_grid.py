import numpy as np

def CreateGrid(rep, OMOPSO):
    
    nGrid = OMOPSO['nGrid']
    alpha = OMOPSO['alpha']
    c   = []      

    for i in range(0, len(rep)):
        c.append(rep[i]['cost'].T)
        
    c_mat = np.array(list(c))        
    
    cmin = c_mat.min(axis=0).T
    cmax = c_mat.max(axis=0).T
    dc = cmax - cmin

    c_min = cmin - alpha*dc
    c_max = cmax + alpha*dc

    nObj = c_min.shape
    nObj = nObj[0]

    c_min = np.array(c_min)
    c_max = np.array(c_max)

    Grid_LB = []
    Grid_UB = []

    init =  -np.inf
    end  =   np.inf

    for j in range(0, nObj):

        cj = np.linspace(c_min[j], c_max[j], nGrid+1)

        Grid_LB.append(np.append(-np.inf, cj))
        Grid_UB.append(np.append(cj, np.inf))

    return Grid_LB, Grid_UB