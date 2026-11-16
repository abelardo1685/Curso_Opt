import numpy as np

def FindGridIndex(rep, Grid_LB, Grid_UB):

    nGrid = len(Grid_LB[0]) # develop the index
    nObj  = len(rep['cost'])

    Grid_Sub_Index = np.zeros((1,nObj))
    vec_cost = rep['cost']

    for j in range(0,nObj):
        Grid_Sub_Index[0,j] = np.argmax(vec_cost[j]<Grid_UB[j])
        
    Grid_Sub_Index =  Grid_Sub_Index+1 # 1: to mimic matlab indexes, otherwise, ERASE THIS LINE   

    rep['grid_subindex'] = Grid_Sub_Index # row vector
    rep['grid_index'] = Grid_Sub_Index[0,0]

    for k in range(1,nObj):
        rep['grid_index'] = rep['grid_index'] -1
        rep['grid_index'] = nGrid * rep['grid_index']
        rep['grid_index'] = rep['grid_index'] + Grid_Sub_Index[0,k] 

    return rep