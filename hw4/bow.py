#==============
from __future__ import print_function

import json
import os
import numpy as np
import pickle
with open('training_label.txt','r',encoding='utf-8') as f:
    trainlist=f.readlines()
with open('testing_data.txt','r',encoding='utf-8') as t:
    testlist=t.readlines()
with open('training_nolabel.txt','r',encoding='utf-8') as t1:
    nllist=t1.readlines()


#str_x=listtostr(list_x)

MAX_NB_WORDS=4000
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
            x=','.join(train[1:len(train)]).lstrip()
        
        list_x.append(x)
        list_y.append(y)
    print( 'len=',len(list_x))    
    return list_x,list_y
    

pre_seq,pre_y=data2vec(testlist,2)    

list_x,list_y=data2vec(trainlist,1)
seqlen=31
'''
list_x = pad_sequences(list_x, maxlen=seqlen,truncating='pre')  
pre_seq = pad_sequences(pre_seq, maxlen=seqlen,truncating='pre')
'''
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(pre_seq + list_x + nllist)
word_index = tokenizer.word_index
#
seq_test = tokenizer.texts_to_sequences(pre_seq)
print('seq_size=',len(seq_test))
seq_train = tokenizer.texts_to_sequences(list_x)
data1 = pad_sequences(seq_train, maxlen=31,truncating='pre')  
test1 = pad_sequences(seq_test, maxlen=31,truncating='pre')
data=[]
pre_data=[]
data.append(tokenizer.texts_to_matrix(pre_seq, mode='binary'))
for i in range(4):
    print(trainlist[i])
    print(data[0])
    print(labels[i])
for i in range(10):
    print('test=',test1[i])
pre_data=tokenizer.sequences_to_matrix(test1, mode='binary')
#===============================================

#=======bulid model==========



labels = to_categorical(np.asarray(list_y))
for i in range(4):
    print(trainlist[i])
    print(data[i])
    print(labels[i])
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


    model=Sequential()
    '''
    model.add(Embedding(len(word_index) + 1,
              EMBEDDING_DIM,
              input_length=seqlen,trainable=False))
    '''

    model.add(Dense(512, input_shape=(max_words,)))
    #                        Embedding(  EMBEDDING_DIM,
     #                       weights=[embedding_matrix],
      #                      input_length=MAX_SEQUENCE_LENGTH,
       #                     trainable=False
    model.add(Dense(512, input_shape=(word_index+1,))) #recurrent_dropout=0.2))
   # model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))

    '''
    model.add(LSTM(
        64,       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
        
        return_sequences=True,      # True: output at all steps. False: output as last step.
        #stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
    ))
    '''
  #  model.add(Flatten())
    #x = MaxPooling1D(35)(x)  # global max pooling
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.5))
  #  model.add(Dense(150, activation='relu'))
   # model.add(Dense(300, activation='relu'))
    model.add(Dense(2, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
                  
    model.summary()              
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=2, batch_size=256,shuffle=True )
              
              
    model.save('model')
else:
    model=load_model('model')
    

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#pre_data = pad_sequences(test, maxlen=seqlen)

out=model.predict(pre_data,verbose=1,batch_size=64)
pre_l=out.argmax(axis=1).reshape(-1,1)
print('shape of pre:',pre_l.shape)
pre_id=np.array(range(0,len(pre_l))).reshape(-1,1)
pre_l=np.concatenate((pre_id,pre_l),axis=1)
np.savetxt('predict.csv',pre_l,fmt='%d',delimiter=',',header='id,label',comments='')
