# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:44:52 2017

@author: USER
"""

import pandas as pd 
import numpy as np
import random
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Embedding ,Dense ,Flatten ,LSTM,Dropout,Activation,Input
from keras.layers import Conv1D, MaxPooling1D ,BatchNormalization
from keras.models import Sequential ,Model
from keras.activations import sigmoid
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.regularizers import  l2,l1
from keras.layers import Bidirectional
from sklearn.svm import SVC

a = open('training_label.txt',encoding='utf8')
List = []
List_label = []
for i in range(200000):
    temp = a.readline()
    temp_t = temp[10:]
    temp_l = temp[0]
    List.append(temp_t)
    List_label.append(temp_l)
    
labels = np.array(List_label,dtype='float64')
labels_1 = to_categorical(labels)

b = open('training_nolabel.txt',encoding='utf8')
nl_List = []
for i in range(1300000):
    temp = b.readline()
    nl_List.append(temp)
    
    
nl_avList=[]
for i in range(len(nl_List)):
    temp = len(nl_List[i])
    if temp < 190:
        nl_avList.append(nl_List[i])    
       

test_List = []
c = open('testing_data.txt',encoding = 'utf8')
for i in range(0,200001):
    temp = c.readline()
    if i == 0:
        continue
    if  1<=i and i<=10:
        temp = temp[2:]
        test_List.append(temp)
        continue
    if i>10 and i<=100:
        temp = temp[3:]
        test_List.append(temp)
        continue
    if i>100 and i <=1000:
        temp = temp[4:]
        test_List.append(temp)
        continue
    if i > 1000 and i<=10000:
        temp=temp[5:]
        test_List.append(temp)
        continue
    if i > 10000 and i<=100000:
        temp = temp[6:]
        test_List.append(temp)
        continue
    if i > 100000 :
        temp = temp[7:]
        test_List.append(temp)
        continue
    
    
def retrain(nl_t):
    c = load_model('80003(public).h5')
    val = c.predict(nl_t,batch_size=1000)
    val = val.reshape(-1)
    record_1 = []
    record_2 = []
    for i in range(len(val)):
        if val[i]>0.95:
            record_1.append(i)
        if val[i]< 0.05  :
            record_2.append(i)
    print(len(record_1))
    print(len(record_2))
    j=0
    for i in record_1:
        temp_1 = nl_t[i,:][np.newaxis,:]
        if j ==0:
            retrain_t = temp_1
            j+=1
            continue
        retrain_t = np.concatenate((retrain_t,temp_1),axis=0)
        print(j)
        j+=1
    j=0
    for i in record_2:
        temp_1 = nl_t[i,:][np.newaxis,:]
        if j ==0:
            retrain_t1 = temp_1
            j+=1
            continue
        retrain_t1 = np.concatenate((retrain_t1,temp_1),axis=0)
        print(j)
        j+=1
    l_1 = [1 for i in range(len(record_1))]
    l_1 = np.array(l_1)
    l_2 = [0 for i in range(len(record_2))]
    l_1 = np.array(l_1)
    nl_label = np.concatenate((l_1,l_2),axis=0)
    nl_t = np.delete(nl_t,record_1 +record_2,axis=0)
    retrain = np.concatenate((retrain_t,retrain_t1),axis=0)
    return nl_t ,retrain ,nl_label

def rd(nl_t):
    m = nl_t.shape[0]
    group = int(m/13)
    List = [i for i in range(m)]
    b = random.sample(List,group)
    j= 0
    for i in b :
        temp = nl_t[i,:][np.newaxis,:]
        if j == 0:
            pre_d = temp
            print(j)
            j +=1
            continue
        pre_d = np.concatenate((pre_d,temp),axis=0)
        print(j)
        j+=1
    nl_t = np.delete(nl_t,b,axis=0)
    prob_l = model.predict(pre_d,batch_size=1000)
    prob_l = prob_l.reshape(-1)
    label_list=[]
    for i in prob_l:
        if i >0.54:
             label_list.append(1)
        else:
            label_list.append(0)
    labels = np.array(label_list)
    return pre_d , labels ,nl_t


def arg(nl_t):
    pp = load_model('8057(2).h5')
    c = pp.predict(nl_t)
    List = []
    List_data = []
    j=0
    for i in range(len(c)):  
        temp = c[i,:]
        maxi = np.argmax(temp)
        if temp[maxi]>0.9:
            List_data.append(i)
            if j == 0:
                List.append(maxi)
                print(j)
                j+=1
                continue
            List.append(maxi)
        print(j)
        j+=1
    nl_l = np.array(List)
    j = 0
    for i in List_data:
        temp = nl_t[i,:][np.newaxis,:]
        if j ==0:
            nl_t2 = temp
            print(j)
            j+=1
            continue
        nl_t2 = np.concatenate((nl_t2,temp),axis=0)
        nl_t = np.delete(nl_t,List_data,axis=0)
        print(j)
        j+=1
    return nl_l ,nl_t2 ,nl_t

total=[]
total += List
total += test_List
tokenizer = Tokenizer()
tokenizer.fit_on_texts(List + test_List + nl_avList)

sequences_t = tokenizer.texts_to_sequences(List)
sequences_test = tokenizer.texts_to_sequences(test_List)
sequences_nl_t = tokenizer.texts_to_sequences(nl_avList)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
train = pad_sequences(sequences_t, maxlen=31)  
test = pad_sequences(sequences_test, maxlen=31)  
nl_t = pad_sequences(sequences_nl_t,maxlen=31)

emb = List + test_List + nl_avList
new_l = []
for i in range(len(emb)):
    temp = text_to_word_sequence(emb[i])
    new_l.append(temp)
model  = Word2Vec(new_l,size=192)
weights = model.wv.syn0
print('weights=',weights.shape)
np.save(open('embeddings.npz', 'wb'), weights)
vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
print('vocab=',len(vocab_list))
embeddings_index = {}
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    coefs = vocab_list[i][1]
    embeddings_index[word]=coefs
    
embeddings_matrix = np.zeros((len(word_index) + 1, 192))
for word, i in word_index.items():
    print('word=',word,'i=',i)
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embeddings_matrix[i] = embedding_vector
    input('enter')
embedding_layer = Embedding(len(word_index)+1,
                            192,
                            weights=[embeddings_matrix],input_length=31,
                            trainable=False)


t_data,v_data,t_label,v_label = train_test_split(train,labels_1,test_size=0.2,shuffle=True,random_state=42)     

model = Sequential()

model.add(embedding_layer)
model.add(Bidirectional(LSTM(192,return_sequences=True,dropout=0.2)))
model.add(Bidirectional(LSTM(192,dropout=0.2)))
model.add(Dense(units=192, activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(2,kernel_initializer='he_normal'))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
history = model.fit(t_data ,t_label, epochs=13, validation_data=(v_data,v_label),shuffle=True ,batch_size=1000,\
                    callbacks=[ModelCheckpoint('best.h5',monitor='val_acc',save_best_only=True)])