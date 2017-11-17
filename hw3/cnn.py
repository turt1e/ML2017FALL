from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import multi_gpu_model
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import sys
def build_model():

    input_img = Input(shape=(48, 48, 1))
 
    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

    block4 = Conv2D(128, (3, 3), activation='relu')(block3)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

    block5 = Conv2D(128, (3, 3), activation='relu')(block4)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Flatten()(block5)

    fc1 = Dense(1024, activation='relu')(block5)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-3)
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model
	

import os
import numpy as np
import argparse
import time
import pickle
import csv
from math import log, floor
import matplotlib.pyplot as plt
os.system('echo $CUDA_VISIBLE_DEVICES')
PATIENCE = 5 # The parameter is used for early stopping

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])
def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    print('len(X_all)=',len(X_all))
    valid_data_size = int(floor(all_data_size * percentage))
    
    X_all, Y_all = _shuffle(X_all, Y_all)
   
    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    
    return X_train, Y_train, X_valid, Y_valid
    
def load_pickle(file):
    f=open(file,'rb')
    data=np.load(f)
    
    return (data)
    
def datatonpy(csvFile,k):
    '''
    trp=open('train_pixels', 'w')
    trl=open('train_labels', 'w')
    vap=open('valid_pixels', 'w')
    val=open('valid_labels', 'w')
    '''
    f=open(csvFile, 'r')
    s=np.loadtxt(f,delimiter=',',skiprows=1,dtype='str')
    l=np.array(s[:,0])
    print (l)
    p=np.zeros((l.shape[0],48*48))
    for i in range(l.shape[0]):
        p[i]=np.array(s[i][1].split(' '))
    print(l.shape[0])
    print(p[0])
    
#        p=np.array(s[:][1])
 #       l=np.array(s[:][0])
        
  #  (tp,tl,vp,vl)=split_valid_set(p, l, 0.1)
  #  np.save('train_pixels.pkl',tp)
  #  pickle.dump(tp, trp)
    if k==0:
        np.save('train_pixels',p)
        np.save('train_labels',l)
   # np.save('valid_pixels',vp)
   # np.save('valid_labels',vl)
    return (l,p)
    #    print(f.shape)
    '''
        for i, line in enumerate(f):
            data = line.split(',')
            label = data[0]
            pixel = data[1]
            l=np.array(label)
			p=np.array(pixel.split(' ')
    '''
              
def main():
    

    epoch=1
    batch=64
    pretrain=True
    save_every=1
    model_name='modelok'
    
    
# ====  load_data  ====  
   # datatonpy('train.csv',0)
#=======================

#    To begin with, you should first read your csv training file and 
#    cut them into training set and validation set.
#   Such as:

#    In addition, we maintain it in array structure and save it in pickle
    
    
    # training data
    train_pixels = load_pickle('train_pixels.npy')
    train_labels = load_pickle('train_labels.npy')
    (train_pixels,train_labels,valid_pixels,valid_labels )=split_valid_set(train_pixels, train_labels, 0.1)
    print ('# of training instances: ' + str(len(train_labels)))
    print(train_labels[1])
    # validation data
   # valid_pixels = load_pickle('valid_pixels.npy')
   # valid_labels = load_pickle('valid_labels.npy')
    print ('# of validation instances: ' + str(len(valid_labels)))
    
    '''
    Modify the answer format so as to correspond with the output of keras model
    We can also do this to training data here, 
        but we choose to do it in "train" function
    '''
    
    valid_pixels = np.reshape(valid_pixels,(-1,48, 48, 1))
    onehot = np.zeros((valid_labels.shape[0],7), dtype=np.float)
    for i in range(len(valid_labels)):
        
        onehot[i][int(valid_labels[i])] = 1.

    valid_labels=onehot    

    # start training
    train(batch, epoch, pretrain, save_every,
          train_pixels, train_labels,
          np.asarray(valid_pixels), np.asarray(valid_labels),
          model_name)
def train(batch_size, num_epoch, pretrain, save_every, train_pixels, train_labels, val_pixels, val_labels, model_name=None):

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    '''
    "1 Epoch" means you have been looked all of the training data once already.
    Batch size B means you look B instances at once when updating your parameter.
    Thus, given 320 instances, batch size 32, you need 10 iterations in 1 epoch.
    '''
 
    train_pixels=np.reshape(train_pixels,(-1,48,48,1))
    train_pixels=train_pixels/255
    
  #  print(train_pixels)
    onehot = np.zeros((train_labels.shape[0],7), dtype=np.float)
    for i in range(len(train_labels)):
        
        onehot[i][int(train_labels[i])] = 1.

    train_labels=onehot 
    
    if pretrain == False:
        np.save('valid_l',val_labels)
        np.save('valid_p',val_pixels/255.0)
        np.save('train_p',train_pixels)
        np.save('train_l',train_labels) 
                
        model=Sequential()

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=(48,48,1),
                         activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
 #       model.add(Dense(30, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.7))    

        model.add(Dense(7, activation='softmax'))
        #opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
        opt = Adam(lr=5e-4)
        #opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)        
        model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
        history=model.fit(train_pixels,train_labels,batch_size=64,epochs=20,validation_data=(val_pixels/255,val_labels))
 
#        model.fit(train_pixels,train_labels,batch_size=32,epochs=3,validation_data=(val_pixels/255,val_labels))
    else:
        model = load_model('modelok')
    
    epochss=3

    datagen = ImageDataGenerator(featurewise_center=False,
                             featurewise_std_normalization=False,
                             shear_range=0.2,
                             
                             zoom_range=0.2
                             )
    datagen.fit(train_pixels)
    

    # compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
  
  # model.fit_generator(datagen.flow(train_pixels, train_labels, batch_size=32),
     #               steps_per_epoch=3000, epochs=8,validation_data=(val_pixels/255,val_labels))


# fits the model on batches with real-time data augmentation:
 # parallel_model = multi_gpu_model(model, gpus=8)
   # for i in range(epochss):
    
 
        
    model.save('modelok')
   
    (no,test)=datatonpy(sys.argc[1],1)
    test=np.reshape(test,(-1,48,48,1))
    np.save('testnp',test)
    test=np.load('testnp.npy')
    y=model.predict(test/255.0)
    yl=np.zeros((len(y),2),dtype=int)
    for i in range(len(y)):
        max=0
        pos=0
        for j in range(7):
            if y[i][j]>max:
                max=y[i][j]
                pos=j
                
        yl[i][0]=i
        yl[i][1]=pos
        
  #  out=open('predict.csv','w')
   # out.write('id,value\n')
    np.savetxt(sys.argc[2],yl,fmt='%d',delimiter=',',header='id,label',comments='')
        
    
    
    
    
    
if __name__=='__main__':
    main()