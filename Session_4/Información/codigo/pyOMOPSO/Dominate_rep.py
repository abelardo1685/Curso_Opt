import numpy as np

def DominateRepository (x,y,e):

    x_vector_s = x['cost'].T
    y_vector   = y['cost'].T
    
    x_vector_s = x_vector_s.flatten()
    y_vector = y_vector.flatten()
    
    x_vector = x_vector_s - e
    
    b  = (x_vector<=y_vector).all() & (x_vector<y_vector).any()

    B = int(b)
    
    return B