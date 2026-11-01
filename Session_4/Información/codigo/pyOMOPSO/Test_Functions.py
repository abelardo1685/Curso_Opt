#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import math 
from abc import ABC, abstractmethod

# Class Test_function_strategy, is a abstract class that helps to develop subclasss for function evaluation  

class Test_function_strategy(ABC):
    @abstractmethod
    def Test_function(self, x) -> "test function evaluation" :
        pass
        
        
class dtlz3(Test_function_strategy):
    def Test_function(self, x) -> 'Function_evaluation':

     #x = [0.3386, 0.6485, 0.5079] #only for testing the code below
     yy = [0, 0, 0]

     vec_x = np.array(x)
     vec_y = np.array(yy)

     g = (vec_x[-1] - 0.5)**2

     yy[0] = (1+g) * math.cos(vec_x[0]*math.pi/2) * math.cos(vec_x[1]*math.pi/2)
     yy[1] = (1+g) * math.cos(vec_x[0]*math.pi/2) * math.sin(vec_x[1]*math.pi/2)
     yy[2] = (1+g) * math.sin(vec_x[0]*math.pi/2) 

     y = np.array([[round(yy[0],4), round(yy[1],4), round(yy[2],4)]]).T
    
     return y
    

class GW_Problem():
    def Two_Objective_function(self, x) -> 'Function_evaluation':

     #x = [0.3386, 0.6485, 0.5079] #only for testing the code below
     yy = [0, 0, 0]

     vec_x = np.array(x)
     vec_y = np.array(yy)

     g = (vec_x[-1] - 0.5)**2

     yy[0] = (1+g) * math.cos(vec_x[0]*math.pi/2) * math.cos(vec_x[1]*math.pi/2)
     yy[1] = (1+g) * math.cos(vec_x[0]*math.pi/2) * math.sin(vec_x[1]*math.pi/2)
     yy[2] = (1+g) * math.sin(vec_x[0]*math.pi/2) 

     y = np.array([[round(yy[0],4), round(yy[1],4), round(yy[2],4)]]).T
    
     return y   


# In[ ]:





# In[43]:


# test 
#x = [0.3386, 0.6485, 0.5079]
#eren = dtlz3()
# leo = eren.Test_function(x)
#print(leo)


# In[44]:





# In[45]:




