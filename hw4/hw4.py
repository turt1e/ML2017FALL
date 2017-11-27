#==============
from __future__ import print_function

import json
import os
import numpy as np
import pickle
with open('training_label.txt','r',encoding='utf-8') as f:
    trainlist=f.readlines()
list_x=[]
list_y=[]
seqlen=0
for i in range(len(trainlist)):
        
    train=trainlist[i].strip('\n').split('+++$+++')
    y=train[0]
    x=train[1]
    length=len(x.split(' '))
    if length>seqlen:
        seqlen=length
        
    list_x.append(x)
    list_y.append(y)
    
maxlen=len(list_x)
print ('len=',len(trainlist))
print ('lenx=',len(list_x))
print('x=',list_x[0])
print('y=',list_y[0])
#===========
def listtostr(list):
    x=np.array([])
    for i in range(2000):
        x=np.append(x,np.array(list_x[i]))
 #       print (x)
    return x
#str_x=listtostr(list_x)
str_x=list_x
MAX_NB_WORDS=20000
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, merge
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from keras.engine import Input

from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding ,LSTM,TimeDistributed
from keras.models import Model

#===============================================
 
#===============================================

#=======bulid model==========

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(str_x)
sequences = tokenizer.texts_to_sequences(str_x)
print(sequences[0])
word_index=tokenizer.word_index
word_size = len(word_index)+1
print('Found %s unique tokens.' % len(word_index))

print('maxseqlen=',seqlen)
data = pad_sequences(sequences, maxlen=seqlen)
labels = to_categorical(np.asarray(list_y))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

nb_validation_samples = int(0.1 * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


MAX_SEQUENCE_LENGTH=seqlen
EMBEDDING_DIM=20
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)



BATCH_SIZE=32
TIME_STEPS=3
INPUT_SIZE=1
train=0
if train==1:
    model=Sequential()
    '''
    model.add(TimeDistributed(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH),batch_input_shape=(32,3,5)))
    '''
    model.add(Embedding(len(word_index) + 1, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    '''
    model.add(LSTM(
        64,       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
        
        return_sequences=True,      # True: output at all steps. False: output as last step.
        #stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
    ))
    '''
    #x = MaxPooling1D(35)(x)  # global max pooling

    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
                  
                  
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=3, batch_size=64)
              
              
    model.save('model')

model=load_model('model')
with open('testing_data.txt','r',encoding='utf-8') as t:
    testlist=t.readlines()
testx=[]
for i in range(1,len(testlist)):
    testseq=testlist[i].split(',')[1]
    testx.append(testseq.strip('\n'))
print(testx[0])
pre_seq=tokenizer.texts_to_sequences(testx)
pre_data = pad_sequences(pre_seq, maxlen=seqlen)
out=model.predict(pre_data)
pre_l=out.argmax(axis=1).reshape(-1,1)
print('shape of pre:',pre_l.shape)
pre_id=np.array(range(0,len(pre_l))).reshape(-1,1)
pre_l=np.concatenate((pre_id,pre_l),axis=1)
np.savetxt('predict.csv',pre_l,fmt='%d',delimiter=',',header='id,label',comments='')
