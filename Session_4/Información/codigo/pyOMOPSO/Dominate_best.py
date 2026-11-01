def dominate_best (x,y):

    x_vector = x['cost'].T
    y_vector = y.T
    
    b  = (x_vector<=y_vector).all() & (x_vector<y_vector).any()

    B = int(b)
    
    return B