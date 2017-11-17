#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 03:08:30 2017

@author: cengbowei
"""

import numpy as np
import pandas as pd
import random
#from PIL import Image
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib
import keras 
from keras.models import load_model
from sklearn.model_selection import train_test_split
'''
from vis.visualization import visualize_activation,visualize_saliency
'''

a = load_model('modelok')
'''
f=open('train.csv', 'r')
s=np.loadtxt(f,delimiter=',',skiprows=1,dtype='str')

label = s[:,0].astype('float64')

t=np.zeros((label.shape[0],48*48))
for i in range(label.shape[0]):
    t[i]=np.array(s[i][1].split(' '))
'''
t=np.load('train_pixels.npy')
label=np.load('train_labels.npy')
t_data,v_data,t_label,v_label = train_test_split(t,label,test_size=0.2,random_state=42,shuffle=True)
r= random.sample([i for i in range(5742)],1)[0]
t_pic = v_data[r]


layer_dict = dict([layer.name, layer] for layer in a.layers[1:])
print(layer_dict['conv2d_2'].output)
nb_class = 7
LR_RATE = 1e-2
NUM_STEPS = 200
RECORD_FREQ = 10
'''
val_proba = a.predict(input_image_data)
pred = val_proba.argmax(axis=-1)
input_img = a.input
target = K.mean(a.output[:, pred])
grads = K.gradients(target, input_img)[0]
fn = K.function([input_img, K.learning_phase()], [grads])
'''
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std()+1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x,0,1)
    x *= 255
    x = np.clip(x,0,255).astype('uint8')
    return x 

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x)))+1e-5)

def grad_ascent(num_step,input_image_data,fn):
    for i in range(num_step):
        loss , grads = fn([input_image_data,0])
        input_image_data += grads * LR_RATE
    return loss ,input_image_data

input_img = a.input
j=0
if j == 0:
    collect_layers=list()
    collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict['conv2d_2'].output]))
    for cnt , fn in enumerate(collect_layers):
        im = fn([t_pic.reshape(1,48,48,1),0])
        fig = plt.figure(figsize=(14,8))
        nb_filter = im[0].shape[3]
        c=0
        for i in range(nb_filter):
            ax = fig.add_subplot(4,16,i+1)
            ax.imshow(im[0][0,:,:,i],cmap='gray')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
            print(c)
            c+=1
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt,17))
    j+=1
if j ==1 :
    name_ls = ['conv2d_2']
    collect_layers = list()
    collect_layers.append(layer_dict[name_ls[0]].output)
    for cnt , c in enumerate(collect_layers):
        filter_imgs = [ [] for i in range(NUM_STEPS//RECORD_FREQ)]
        filter_imgs_2 = [ [] for i in range(NUM_STEPS//RECORD_FREQ)]
        nb_filter = c.shape[-1]
        for filter_idx in range(nb_filter):
            input_image_data = np.random.random((1,48,48,1))
            count = 0
            for f in range(10,210,10):
                loss = K.mean(c[:,:,:,filter_idx])
                grads = normalize(K.gradients(loss,input_img)[0])
                iterate = K.function([input_img,K.learning_phase()],[loss,grads])
                temp_l,input_image_data = grad_ascent(10,input_image_data,iterate)
                '''
                List.append(temp_g)
                loss_t.append(temp_l)
                '''
                filter_imgs[count].append(input_image_data)
                filter_imgs_2[count].append(temp_l)
                print(count)
                count+=1
            '''
            filter_imgs[count].append(List)
            filter_imgs_2[count].append(loss_t)
            '''
        
        for it in range(NUM_STEPS//RECORD_FREQ):
            t= 0
            fig = plt.figure(figsize=(14,8))
            for i in range(50):
                ax = fig.add_subplot(6,16,i+1)
                temp_p=filter_imgs[it][i][0].squeeze()
                ax.imshow(temp_p,cmap='gray')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs_2[it][i]))
                plt.tight_layout()
                print(t)
                t+=1
            fig.suptitle('Filters of layer {} (# Ascent Epoch {})'.format(name_ls[0],RECORD_FREQ))
                    

'''
b = visualize_activation(a,layer_idx=0,filter_indices=None)
layer1_output = K.function([a.input,K.learning_phase()],a.layers[0].output)
layer_output = layer1_output([b,0])
'''
