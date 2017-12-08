#==============
from __future__ import print_function
import sys
import json
import os
import numpy as np
import pickle
with open(sys.argv[1],'r',encoding='utf-8') as f:
    trainlist=f.readlines()
'''    
with open('testing_data.txt','r',encoding='utf-8') as t:
    testlist=t.readlines()
'''    

with open(sys.argv[2],'r',encoding='utf-8') as t1:
    nllist=t1.readlines()


#str_x=listtostr(list_x)
from keras.callbacks import ModelCheckpoint
MAX_NB_WORDS=40000
from keras.models import load_model
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
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

#===============================================
#================================
#============word2vec============
#================================



    
# specify embeddings in this environment variable
data_path = 'train'
'''
# variable arguments are passed to gensim's word2vec model
create_embeddings(data_path, size=200, min_count=3,
                  window=3, sg=1, iter=2)
'''


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
    
 

list_x,list_y=data2vec(trainlist,1)
all_x=list_x+nllist


seqlen=31


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


word_index = tokenizer.word_index

seq_train = tokenizer.texts_to_sequences(list_x)
data = pad_sequences(seq_train, maxlen=31,truncating='pre')  



# loading


#===============================================

#=======bulid model==========



labels = to_categorical(np.asarray(list_y))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

nb_validation_samples = int(0.1 * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


MAX_SEQUENCE_LENGTH=seqlen
EMBEDDING_DIM=100
from keras.layers import Embedding

all = []
for i in range(len(all_x)):
    temp = text_to_word_sequence(all_x[i])
    all.append(temp)



modelwv  = Word2Vec(all,size=192)
modelwv.save('w21vmodel')

modelwv=Word2Vec.load('w2vmodel')
weights = modelwv.wv.syn0
print('weights.shape[0]=',weights.shape[0])

vocab_list = [(k, modelwv.wv[k]) for k, v in modelwv.wv.vocab.items()]
print('vocab=',len(vocab_list))
embeddings_index = {}
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    coefs = vocab_list[i][1]
    embeddings_index[word]=coefs

embeddings_matrix = np.zeros((len(word_index) + 1, 192))
for word, i in word_index.items():
  #  print('word=',word,'i=',i)
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embeddings_matrix[i] = embedding_vector
  #  input('enter')  
train=1
if train==1:
    embedding_layer = Embedding(len(word_index)+1,
                            192,
                            weights=[embeddings_matrix],input_length=31,
                            trainable=False)


    model = Sequential()

    model.add(embedding_layer)
    model.add(LSTM(192,dropout=0.2,return_sequences=True))
    model.add(LSTM(192,dropout=0.2))
    model.add(Dense(units=192, activation='relu',kernel_initializer='he_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(2,kernel_initializer='he_normal',activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])                 
    model.summary()              
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=6, batch_size=256,shuffle=True ,callbacks=[ModelCheckpoint('model',monitor='val_acc',save_best_only=True)])
              
              
else:
    model=load_model('model')
    



