#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def erase_dominated(pop):

    rep = []

for i in range(0, len(pop)):
    if pop[i]['is_dominated'] == 1:
        rep.append(pop[i]['is_dominated'])
        
    return rep
        
        

