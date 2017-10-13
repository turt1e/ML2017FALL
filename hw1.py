
# coding: utf-8

# In[403]:


import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys


# In[404]:


# save model
#np.save('model.npy',w)

# read model
w = np.load('model.npy')


# In[405]:


test_x = []
n_row = 0
text = open('data/test.csv' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if (n_row %18 == 0):
        test_x.append([])
        for i in range(6,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
 #   if n_row %18 == 9:
        for i in range(6,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()

test_x = np.array(test_x)

# add square term
#test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)


# In[ ]:





# In[406]:


ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = "result/predict.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()


# In[407]:


np.shape(test_x[i])
np.shape(w)

