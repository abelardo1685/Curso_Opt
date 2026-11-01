def Dominate_pop (x,y):

    #x_vector = x['cost'].T
    x_vector = x.T
    y_vector = y['cost'].T

    b  = (x_vector<=y_vector).all() & (x_vector<y_vector).any()

    B = int(b)
    
    return B
