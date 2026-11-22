import numpy as np
from Det_Dom_Rep import DetermineDominationRepository

def Repository_dominance(rep,e_file,e):

    ## Continuos update of the repository of non dominant f vectors.

    # rep: Matriz de vectores no dominantes
    # f: current objective vector used for comparison

    #    keyboard
    ## Algoritmo 2: Marco Laumanns: Combining Convergence and Dispersivity in
    # Evolutionary Multiobjective Optimization.

    # Determina dominacion a traves de e-values Pareto comparison using e
    # efficiency values
    for i in range(0, len(rep)):
        e_file.append(rep[i])

    # Use of e efficiency values
    # Determine Domination of New Resository Members
    e_file = DetermineDominationRepository(e_file,e)

    # Segunda parte donde se miden las dimensiones de los hypercubos.    
    ## Algoritmo 3: box
    # Box genera las dimensiones de un hypercubo por particula

    ## Algoritmo 4:  Update e-pareto set
    # Analisis comparativo por particula del repositorio vs con particula
    # seleccionada.
    nPop  = len(e_file) # tamano del repositorio   
    den   = np.log(1+e)
    
    for i in range(0,nPop-1):
        for j in range(i + 1, nPop):
        
        # Calculo de indeces de caja
            cost_i     =  e_file[i]['cost'].copy();
            cost_j     =  e_file[j]['cost'].copy();
            
            if np.any(cost_i==0):
                cost_i = cost_i + e
            
            if np.any(cost_j==0):
                cost_j = cost_j + e
            

            box_i = np.log(cost_i) / den
            box_j = np.log(cost_j) / den
            
             ## Dominancia para encontrar el minimo global
                
            # Primera comparacion: Si las medidas de box son parecidas.
            if (box_i<=box_j).all() & (box_i<box_j).any():
                e_file[j]['is_dominated'] = 1
                
            # Segunda comparacion: Si las medidas de box son iguales pero los
            # valores de la nueva particula son mejores, borra la particula
            # anterior (comparacion de misma casilla)
            elif (box_i==box_j).all() & ((cost_i<=cost_j).all() & (cost_i<cost_j).any()).all():
                e_file[j]['is_dominated'] = 1
                
    # Borrando las particulas previas.
    e_file = [sub for sub in e_file if sub['is_dominated'] == 0]

    return e_file, rep