from gensim.models import word2vec
import pandas as pd 
import numpy as np
import csv

import sys
import os
import argparse
import _pickle as pk
import jieba


jieba.set_dictionary('dict.txt.big')

def chineseCut(filename):
    data = []
    with open(filename,'r',encoding = 'utf8') as content:
        for row in content:
            row = row.strip()

            word=jieba.cut(row, cut_all=False)
            word1=" ".join(word)
            data.append(word1)
   #         print (type(word1))
  #          print ((word1))
            
 
        

    return data

def loadTestData(filename):
    with open(filename,'r') as f:
        f=f.readlines()
        question = []
        option1 = []
        option2 = []
        option3 = []
        option4 = []
        option5 = []
        option6 = []


        for row in f:
            row = row.strip()
            iden, qu, op = row.split(',')
            tmpqu = []
            word =jieba.cut(qu, cut_all=False)
            print(len(word))
            tmpqu.append(word)
            print(word)
              
            question.append(tmpqu)
            op1, op2, op3, op4, op5, op6 = op.split('\t')
            tmpop1 = []
            tmpop2 = []
            tmpop3 = []
            tmpop4 = []
            tmpop5 = []
            tmpop6 = []
            for word in jieba.cut(op1, cut_all=False):
                tmpop1.append(word)
            for word in jieba.cut(op2, cut_all=False):
                tmpop2.append(word)
            for word in jieba.cut(op3, cut_all=False):
                tmpop3.append(word)
            for word in jieba.cut(op4, cut_all=False):
                tmpop4.append(word)
            for word in jieba.cut(op5, cut_all=False):
                tmpop5.append(word)
            for word in jieba.cut(op6, cut_all=False):
                tmpop6.append(word)

            option1.append(tmpop1)
            option2.append(tmpop2)
            option3.append(tmpop3)
            option4.append(tmpop4)
            option5.append(tmpop5)
            option6.append(tmpop6)
        



    return question ,option1, option2, option3, option4, option5, option6

def construtVector(sen_list,models):
    
    question_vec = []
    for rows in sen_list:
        temp = np.zeros(250)
        for ele in rows:
            if ele in models.wv.vocab:
                temp = temp+np.array(models.wv[ele])
            else:
                temp = temp
                
        question_vec.append(temp.tolist())
    return question_vec
def computeSimilarity(vec1,vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    simi = np.dot(vec1,vec2)
    return simi

data_train1 = chineseCut('1_train.txt')
data_train2 = chineseCut('2_train.txt')
data_train3 = chineseCut('3_train.txt')
data_train4 = chineseCut('4_train.txt')
data_train5 = chineseCut('5_train.txt')



data_all = data_train1+data_train2+data_train3+data_train4+data_train5

l=len(data_all)
#===========data preprocessing


from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, LSTM, Flatten
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot, Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_all )
word_index = tokenizer.word_index
seq_train = tokenizer.texts_to_sequences(data_all )
print('seq_size=',len(seq_train))

training_data= pad_sequences(seq_train, maxlen=10,truncating='pre')  
print(training_data[0])



frontback=[]
uncor=[]
idx=0
coef=np.array([])
for i in range(l-20):
    tem=np.array([])
    tem=np.concatenate(training_data[i],training_data[i+1])
    frontback.append(tem)
    coef=np.append(1)
    tem1=np.concatenate(training_data[i],training_data[i+10])
    frontback.append(tem1)
    coef=np.append(0)
    

frontback

go=1
if go==1:
    embedding_layer1 = Embedding(len(word_index)+1,
                            30,
                            input_length=10,
                            trainable=True,name='user_em')
    embedding_layer2 = Embedding(len(word_index)+1,
                            30,
                            input_length=10,
                            trainable=True,name='movie_em')    


    u=Input(shape=(10,),name='user')
    m=Input(shape=(10,),name='movie')

    r=embedding_layer1(u)
    p=embedding_layer2 (m)

    
    r1=LSTM(30,)(r)
    p1=LSTM(30,)(p)
    
    dot=Dot(axes=1,name='dot')([r1,p1])
    model=Model(inputs=[u,m],outputs=[dot])
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
                  
    model.fit([frontback[:l*0.7][0],frontback[:l*0.7][1]], coef[:l*0.7], validation_data=([frontback[l*0.7:][0],frontback[l*0.7:][1]], coef[l*0.7:]),
              epochs=8, batch_size=256,)
              


question_test ,option1_test, option2_test, option3_test, option4_test, option5_test, option6_test= loadTestData('testing_data.csv')



# models = word2vec.Word2Vec(data_all, size=250, min_count=5, workers=4)
# models.save('modwtov_fi250.model')
# words = sorted(models.wv.vocab.keys())
# print("Number of words:", len(words))

# print(models.similarity('我','他'))
# print(models.wv['狗'])
# sent = [['我'],['是']]

# for i in sent:
#     print(models.wv[i])

models = word2vec.Word2Vec.load('modwtov_fi250.model')

qu_vec = construtVector(question_test,models)
op1_vec = construtVector(option1_test,models)
op2_vec = construtVector(option2_test,models)
op3_vec = construtVector(option3_test,models)
op4_vec = construtVector(option4_test,models)
op5_vec = construtVector(option5_test,models)
op6_vec = construtVector(option6_test,models)



si0 = 0
si1 = 0
si2 = 0
si3 = 0
si4 = 0
si5 = 0
ans = []
print(len(qu_vec))

for i in range(len(qu_vec)):
    ans.append([str(i+1)])
    si0 = computeSimilarity(qu_vec[i],op1_vec[i])
    si1 = computeSimilarity(qu_vec[i],op2_vec[i])
    si2 = computeSimilarity(qu_vec[i],op3_vec[i])
    si3 = computeSimilarity(qu_vec[i],op4_vec[i])
    si4 = computeSimilarity(qu_vec[i],op5_vec[i])
    si5 = computeSimilarity(qu_vec[i],op6_vec[i])
    print(si0)
    print(si1)
    print(si2)
    print(si3)
    print(si4)
    print(i)

    if max(si0,si1,si2,si3,si4,si5)==si0:
        a=0
        
    elif max(si0,si1,si2,si3,si4,si5)==si1: 
        a=1
        
    elif max(si0,si1,si2,si3,si4,si5)==si2: 
        a=2
        
    elif max(si0,si1,si2,si3,si4,si5)==si3: 
        a=3
        
    elif max(si0,si1,si2,si3,si4,si5)==si4: 
        a=4
        
    elif max(si0,si1,si2,si3,si4,si5)==si5: 
        a=5
        
    ans[i].append(a)

filename = 'prediction_ficon.csv'
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","ans"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()




























