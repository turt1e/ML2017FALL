from __future__ import print_function
import sys
import json
import os
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, merge,Dropout
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from keras.engine import Input
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding ,LSTM,TimeDistributed
from keras.models import Model

def data2vec(trainlist,mode):    
    seqlen=0
    list_x=[]
    list_y=[]
    if mode==2:
        trainlist=trainlist[1:len(trainlist)]
        print ('trl=',len(trainlist))
    for i in range(len(trainlist)):
        if mode==1:    
            train=trainlist[i].strip('\n').split('+++$+++')
        if mode==2:
                
      #      print(trainlist[i][0])
            train=trainlist[i].strip('\n').split(',')
            if i==0:
                print(train[1])
        y=train[0]
        x=train[1].lstrip()
        if mode==2:
            x=','.join(train[1:len(train)])
        
        list_x.append(x)
        list_y.append(y)
    print( 'len=',len(list_x))    
    return list_x,list_y
   
   
   
with open(sys.argv[1],'r',encoding='utf-8') as t:
    testlist=t.readlines()

pre_seq,pre_y=data2vec(testlist,2)  
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

tokenizer.fit_on_texts(pre_seq)

seq_test = tokenizer.texts_to_sequences(pre_seq)
test = pad_sequences(seq_test, maxlen=31,truncating='pre')
model=load_model('model')
    


out=model.predict(test,verbose=1,batch_size=1000)

print(out)
pre_l=out.argmax(axis=1).reshape(-1,1)
print('shape of pre:',pre_l.shape)
pre_id=np.array(range(0,len(pre_l))).reshape(-1,1)
pre_l=np.concatenate((pre_id,pre_l),axis=1)
np.savetxt('predict.csv',pre_l,fmt='%d',delimiter=',',header='id,label',comments='')
