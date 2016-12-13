# -*- coding: utf-8 -*-
import numpy as np
from keras.layers.advanced_activations import SReLU
from keras import callbacks as ckbs
import random
import time
import gzip
from keras.models import model_from_json
import pickle as pkl
import keras
from keras.models import Sequential
from keras.layers.core import *
from keras.layers.wrappers import *
from keras.optimizers import *
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from docutils.languages.af import labels
from sklearn.model_selection import train_test_split


def load_neural_network(file_from):
    (nn_arch, nn_weights_path) = pkl.load(open(file_from, 'rb'))
    nn = model_from_json(nn_arch)
    nn.set_weights(nn_weights_path)
    return nn

f = gzip.open('../data/processed/pkl/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

label2Idx = embeddings['label2Idx']
wordEmbeddings = embeddings['wordEmbeddings']

#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}

f = gzip.open('../data/processed/pkl/data.pkl.gz', 'rb')
test_data = pkl.load(f)
dev_data = pkl.load(f)
wiki_data = pkl.load(f)
f.close()

n_out = len(label2Idx)
  

print( "%d test sentences" % len(wiki_data))
wiki = np.array(wiki_data)
print(wiki.shape)
X = [x[0][0] for x in wiki_data]
y = [x[1][0] for x in wiki_data]
#print(X)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('result')
model = load_neural_network('../models/best_model')
pred = model.predict_classes(X_test, batch_size=128, verbose=1)
ff = open('2.txt', 'w+')
for x in pred:
    ff.write(str(x))
#print(pred)
