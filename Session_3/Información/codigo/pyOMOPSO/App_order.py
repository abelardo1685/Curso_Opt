def appearance_sort(x):
   # a = sorted(x, key=lambda y: x.index(y), reverse=True)

    b = sorted(range(len(x)), key=lambda k: x[k])

    return b