
import sys
print(sys.argv[1])
print(sys.argv[2])
import numpy as np
testfile=sys.argv[1]
predictfile=sys.argv[2]



test=np.loadtxt(testfile,delimiter =',',skiprows=1)
#train=np.loadtxt(sys.argv[2],delimiter =',',skiprows=1)


from keras.models import load_model
from keras.regularizers import l2
from keras.layers import Dense
from keras.layers import Flatten, merge
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.engine import Input

from keras.layers import Dense, Input, Flatten,Dropout,Dot,Concatenate,Add
from keras.models import Model

go=0


model=load_model('model')
out=model.predict([test[:,1],test[:,2]])

#np.clip(a, 1, 8)
#outy=out
outy=np.clip(out,0,5)

for i in range(len(outy)):
    if outy[i]>5:
        print('i=',outy[i])
with open(predictfile,'w') as o:
    o.write('TestDataID,Rating\n')
    for i in  range(len(outy)):
        o.write('%d,%f\n' %(i+1,outy[i] ))
