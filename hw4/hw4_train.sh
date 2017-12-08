#!/bin/bash 
wget -O w2vmodel.syn1neg.npy https://www.dropbox.com/s/wpdm4e2n1duelgw/w2vmodel.syn1neg.npy?dl=0
wget -O w2vmodel.wv.syn0.npy https://www.dropbox.com/s/qe2t64g10fhuux9/w2vmodel.wv.syn0.npy?dl=0
python hw4train.py $1 $2
