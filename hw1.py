
# coding: utf-8

# In[220]:


import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import codecs


# In[221]:


data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))
    n_row = n_row+1
 #   print (r)
text.close()


x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
 #       for t in range(8,10):
            # 連續9小時
        for s in range(6):
            x[471*i+j].append(data[9][480*i+j+s] )
        y.append(data[9][480*i+j+6])
x = np.array(x)
y = np.array(y)

# add square term
x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)


# In[249]:


w = np.zeros(len(x[0]))
l_rate = 30
lamda=2
repeat = 10000


# In[250]:





# In[251]:


x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))


# In[252]:

# In[253]:


# save model
np.save('model.npy',w)

# read model
w = np.load('model.npy')


# In[254]:


test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if (n_row %18 == 9):
        test_x.append([])
    #    for i in range(7,11):
    #        test_x[n_row//18].append(float(r[i]) )
  #  else :
  #  if n_row %18 == 9:
        for i in range(5,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()

test_x = np.array(test_x)

# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)


# In[ ]:





# In[255]:


ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()


# In[256]:


np.shape(test_x[i])
np.shape(w)


