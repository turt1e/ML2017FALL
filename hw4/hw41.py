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
#================================
#============word2vec============
#================================


# tokenizer: can change this as needed
tokenize = lambda x: simple_preprocess(x)


def create_embeddings(data_dir,
                      embeddings_path='embeddings.npz',
                      vocab_path='map.json',
                      **params):
    """
    Generate embeddings from a batch of text
    :param embeddings_path: where to save the embeddings
    :param vocab_path: where to save the word-index map
    """

    class SentenceGenerator(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            for fname in os.listdir(self.dirname):
                print (fname)
                for line in open(os.path.join(self.dirname, fname),encoding='utf-8'):
                    yield tokenize(line)

    sentences = SentenceGenerator(data_dir)

    model = Word2Vec(sentences, **params,batch_words=20000)
    weights = model.wv.syn0
    print('weights=',weights.shape)
    np.save(open(embeddings_path, 'wb'), weights)

    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))


def load_vocab(vocab_path='map.json'):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """

    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


def word2vec_embedding_layer(embeddings_path='embeddings.npz'):
    """
    Generate an embedding layer word2vec embeddings
    :param embeddings_path: where the embeddings are saved (as a numpy file)
    :return: the generated embedding layer
    """
    
    weights = np.load(open(embeddings_path, 'rb'))
    layer = Embedding(input_dim=weights.shape[0],
                      output_dim=weights.shape[1],
                      weights=[weights])
    return layer

    
# specify embeddings in this environment variable
data_path = 'train'
'''
# variable arguments are passed to gensim's word2vec model
create_embeddings(data_path, size=100, min_count=5,
                  window=3, sg=1, iter=2)
'''
word2idx, idx2word = load_vocab()

# cosine similarity model
input_a = Input(shape=(1,), dtype='int32', name='input_a')
input_b = Input(shape=(1,), dtype='int32', name='input_b')
embeddings = word2vec_embedding_layer()
embedding_a = embeddings(input_a)
embedding_b = embeddings(input_b)
similarity = merge([embedding_a, embedding_b],
                   mode='cos', dot_axes=2)

modelwv = Model(input=[input_a, input_b], output=[similarity])
modelwv.compile(optimizer='sgd', loss='mse')

for i in range(0):
    word_a = input('First word: ')
    if word_a not in word2idx:
        print('Word "%s" is not in the index' % word_a)
        continue
        
   # word_b = input('Second word: ')
    '''
    if word_b not in word2idx:
        print('Word "%s" is not in the index' % word_b)
        continue
    output = model.predict([np.asarray([word2idx[word_a]]
    ),
                            np.asarray([word2idx[word_b]])]
    '''         
    print (type(word_a))
    print([word2idx[word_a]])
def data2vec(trainlist,word2idx,mode):    
    seqlen=0
    list_x=[]
    if mode==2:
        trainlist=trainlist[1:len(trainlist)]
    id=len(word2idx)
    print('word2idx=',len(word2idx))
    for i in range(len(trainlist)):
        if mode==1:    
            train=trainlist[i].strip('\n').split('+++$+++')
        if mode==2:
            train=trainlist[i].strip('\n').split(',')
            
        y=train[0]
        x=train[1].lstrip()
        if mode==2:
            x=','.join(train[1:len(train)])
        length=len(x.split(' '))
        if length>seqlen:
            seqlen=length
        x=tokenize(x)
       # print(x)
        xid=[]
     #   print(type(str1))   
        for j in range(len(x)):
          #  if i==0:
         #       print(x[j],word2idx[x[j]])
            if x[j] not in word2idx:
          #      print('Word "%s" is not in the index' % x[j])
                xid.append(id)
            else:
                xid.append(word2idx[x[j]])

        list_x.append(xid)
    #    list_y.append(y)
    print( list_x[0])    
    return list_x,seqlen
#sequences,seqlen=data2vec(trainlist,word2idx,1)
#===============================================

#=======bulid model==========

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(str_x)
sequences = tokenizer.texts_to_sequences(str_x)
print(sequences[0])
word_index=tokenizer.word_index
word_size = len(word_index)+1
print('Found %s unique tokens.' % len(word_index))
#input("Press Enter to continue...")
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
EMBEDDING_DIM=200
from keras.layers import Embedding
'''
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)
'''


BATCH_SIZE=32
TIME_STEPS=3
INPUT_SIZE=1
train=1

if train==1:
    weights = np.load(open('embeddings.npz', 'rb'))
    print(weights.shape)
    print(type(weights))
    x=np.zeros((1,weights.shape[1]))
    weights=np.append(weights,x,axis=0)
    print(weights[len(word2idx)])
    model=Sequential()
    
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH))
    
    '''
    model.add(Embedding(input_dim=weights.shape[0],
                      output_dim=weights.shape[1],
                      weights=[weights],trainable=False))
    '''
       #                     Embedding(  EMBEDDING_DIM,
        #                    weights=[embedding_matrix],
         #                   input_length=MAX_SEQUENCE_LENGTH,
          #                  trainable=False
    model.add(LSTM(256, dropout=0.2))
    '''
    model.add(LSTM(
        64,       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
        
        return_sequences=True,      # True: output at all steps. False: output as last step.
        #stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
    ))
    '''
    #x = MaxPooling1D(35)(x)  # global max pooling

    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
                  
                  
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=8, batch_size=256)
              
              
    model.save('model')

model=load_model('model')
with open('testing_data.txt','r',encoding='utf-8') as t:
    testlist=t.readlines()
testx=[]
#pre_seq,seqlen=data2vec(testlist,word2idx,2)

for i in range(1,len(testlist)):
    testseq=testlist[i].split(',')
    testseq=testseq[1:len(testseq)]
    testseq=','.join(testseq)
    testx.append(testseq.strip('\n'))
print(testx[0])
pre_seq=tokenizer.texts_to_sequences(testx)

pre_data = pad_sequences(pre_seq, maxlen=seqlen)

out=model.predict(pre_data,batch_size=1000)
pre_l=out.argmax(axis=1).reshape(-1,1)
print('shape of pre:',pre_l.shape)
pre_id=np.array(range(0,len(pre_l))).reshape(-1,1)
pre_l=np.concatenate((pre_id,pre_l),axis=1)
np.savetxt('predict.csv',pre_l,fmt='%d',delimiter=',',header='id,label',comments='')
