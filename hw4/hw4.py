#==============
import numpy as np

with open('training_label.txt','r',encoding='utf-8') as f:
    trainlist=f.readlines()
list_x=[]
list_y=[]
for i in range(len(trainlist)):
        
    train=trainlist[i].strip('\n').split(' ')
    y=train[0]
    x=train[2:len(train)]
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
str_x=listtostr(list_x)
MAX_NB_WORDS=20000

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(str_x)
sequences = tokenizer.texts_to_sequences(str_x)
 #   print(i)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print('Found %s unique words.' % len(str_x))
print(x)
data = pad_sequences(sequences, maxlen=5000)


