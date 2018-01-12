import pickle
import numpy as np
import sys

testcase=sys.argv[2]
print('test=',testcase)
output=sys.argv[3]
print('output=',output)
#predict

label=np.loadtxt(testcase,skiprows=1,delimiter=',' )

predict=np.zeros((label.shape[0],2))
#k_label=kmeans.labels_
k_label=np.load('k_label.npy')

for i in range(label.shape[0]):
    predict[i][0]=i
    pos1=int(label[i][1])
    
    pos2=int(label[i][2])
  #  print(pos1)
    x=[k_label[pos1], k_label[pos2]]

    if x[0]==x[1]:
        predict[i][1]=1
    else:
        predict[i][1]=0
    if i<10:
        print([x[0], x[1]])
        print(predict[i][1])
np.savetxt(output,predict,delimiter=',',fmt='%d',comments='',header='ID,Ans')
