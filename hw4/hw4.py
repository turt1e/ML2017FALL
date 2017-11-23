import numpy as np

with open('training_label.txt','r',encoding='utf-8') as f:
    trainlist=f.readlines()
list_x=[]
list_y=[]
for i in range(len(trainlist)):
        
    train=trainlist[i].split(' ')
    y=train[0]
    x=train[2:-1]
    list_x.append(x)
    list_y.append(y)
    
    
print ('len=',len(trainlist))
print ('lenx=',len(list_x))
print('x=',list_x[0])
print('y=',list_y[0])