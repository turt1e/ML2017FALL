import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd

# If you wish to get the same shuffle result
# np.random.seed(2401)

def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)
  #  X_test=np.concatenate((X_test,X_test**2),axis=1)
 #   X_test=np.concatenate((X_test,X_test**3),axis=1)
  #  X_train=np.concatenate((X_train,X_train**2,),axis=1)
 #   X_train=np.concatenate((X_train,X_train**3),axis=1)
    A=[0,1,2,3,4,5,11,13,15,19,28,31,33,34,35,39,46,53,54,56,58,64,66,67,68,71,73,74,76,77,80,81,83,84,86,87,88,93,95,97,99,102,103]
    X_train=np.concatenate((X_train[:,A],X_train**2),axis=1)
    X_test =np.concatenate((X_test[:,A],X_test**2),axis=1)
    A=np.arange(31,38)
    A1=np.arange(53,100)
    A=np.concatenate((A,A1),axis=0)
  #  X_train=np.delete(X_train,A,axis=1)
  #  X_test=np.delete(X_test,A,axis=1)
  #  X_train=np.delete(X_train,[6,9,10,38,48,50,51,53,76],axis=1)
  #  X_test=np.delete(X_test,[6,9,10,38,48,50,51,53,76],axis=1)
  #  X_train=np.delete(X_train,[11,19,43,45,66,69],axis=1)
  #  X_test=np.delete(X_test,[11,19,43,45,66,69],axis=1)
  #  X_train=np.delete(X_train,[6,8,10,11,29,31,64,89],axis=1)
  #  X_test=np.delete(X_test,[6,8,10,11,29,31,64,89],axis=1)
    #print(A)
    return (X_train, Y_train, X_test)

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    print(X_train_test.shape[1])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    np.savetxt('X_all.csv', X_all , delimiter=',')
    np.savetxt('X_test.csv', X_test, delimiter=',')
    return X_all, X_test

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def sigmoid(z):
    #print('z=%d,',z[1])
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    return float(result.sum()) / valid_data_size

def train(X_all, Y_all, save_dir):

    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)

    # Initiallize parameter, hyperparameter
    w = np.zeros((len(X_all[0]),))
    b = np.zeros((1,))
    l_rate = 1
    batch_size = 100
    train_data_size = len(X_train)
    step_num = int(floor(train_data_size / batch_size))
    epoch_num = 1000
 #  lamda=3
    save_param_iter = 100
    ada=0
    max_valid=0.0
    ada_b=0
    # Start training
    total_loss = 0.0
    for epoch in range(1, epoch_num):

#        w=np.random.randn(len(X_all[0]),)
#        w=w/np.sum(w)

        # Do validation and parameter saving
        if (epoch) % save_param_iter == 0:
            print('=====Saving Param at epoch %d=====' % epoch)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, 'w'), w)
            np.savetxt(os.path.join(save_dir, 'b'), [b,])
            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
            valid(w, b, X_valid, Y_valid)
        # Random shuffle
        X_train, Y_train = _shuffle(X_train, Y_train)
        

        # Train with batch
        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]
           # print(X)
            z = np.dot(X, np.transpose(w)) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            total_loss += cross_entropy

            w_grad = np.sum(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0)#+2*lamda*w
            b_grad = np.sum(-1 * (np.squeeze(Y) - y))
            ada=ada+w_grad**2
            ada_b=ada_b+b_grad**2
            # SGD updating parameters
            w = w - l_rate * w_grad/np.sqrt(ada)
            b = b - l_rate * b_grad/np.sqrt(ada_b)

    return

def infer(X_test, save_dir, output_path):
    test_data_size = len(X_test)

    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    w = np.loadtxt(os.path.join(save_dir, 'w'))
    b = np.loadtxt(os.path.join(save_dir, 'b'))

    # predict
    z = (np.dot(X_test, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)

    print('=====Write output to %s =====' % output_path)

    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return

def main():
    # Load feature and label
    save_dir='logistic_params/'
    X_all, Y_all, X_test = load_data(sys.argv[4], sys.argv[5], sys.argv[6])
    # Normalization
    X_all, X_test = normalize(X_all, X_test)

    # To train or to infer

    train(X_all, Y_all, save_dir)

    infer(X_test, save_dir, sys.argv[7])


    return
if __name__ == "__main__":
    main()


