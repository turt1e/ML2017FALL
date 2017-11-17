#!/usr/bin/env python
# -- coding: utf-8 --
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from utils import *
import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def read_dataset(data_path):
    train_pixels = np.load(data_path)
#    for i in range(len(train_pixels)):
#        train_pixels[i] = np.fromstring(train_pixels[i], dtype=float, sep=' ').reshape((48, 48, 1))
    return np.asarray(train_pixels)

def get_labels(data_path):
    train_labels = load_pickle(data_path)
    train = []
    for i in range(len(train_labels)):
        train.append(int(train_labels[i]))
    return np.asarray(train)

def main():
    model_path = 'modelok'
    emotion_classifier = load_model(model_path)
    np.set_printoptions(precision=2)
    dev_feats = read_dataset('valid_p.npy')
    predictions = emotion_classifier.predict(dev_feats)
    predictions = predictions.argmax(axis=-1)
    print (predictions)
    te_labels = np.load('valid_labels.npy')
    print (te_labels)
    conf_mat = confusion_matrix(np.int_(te_labels),predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    plt.show()

if __name__=='__main__':
    main()