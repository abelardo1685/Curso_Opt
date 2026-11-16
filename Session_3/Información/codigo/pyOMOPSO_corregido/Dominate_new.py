def dominate_new (x,y):

    x_vector = x.T
    y_vector = y['cost'].T
    

    b  = (x_vector<=y_vector).all() & (x_vector<y_vector).any()

    B = int(b)
    
    return B