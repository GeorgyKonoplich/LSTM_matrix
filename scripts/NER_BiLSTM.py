# -*- coding: utf-8 -*-
import numpy as np
import random
import time
import gzip
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


f = gzip.open('../data/processed/pkl_short/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

label2Idx = embeddings['label2Idx']
wordEmbeddings = embeddings['wordEmbeddings']

#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}

f = gzip.open('../data/processed/pkl_short/data.pkl.gz', 'rb')
test_data = pkl.load(f)
dev_data = pkl.load(f)
wiki_data = pkl.load(f)
f.close()



#####################################
#
# Create the  Network
#
#####################################

n_out = len(label2Idx)
  
model = Sequential()
model.add(Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False))
model.add(LSTM(10, dropout_W=0.2)) 
model.add(Dense(n_out, activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-7, momentum=0.0, nesterov=False, clipvalue=3) 
rmsprop = RMSprop(clipvalue=3) 
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.summary()


##################################
#
# Training of the Network
#
#################################

print( "%d test sentences" % len(wiki_data))
wiki = np.array(wiki_data)
print(wiki.shape)
X = [x[0][0] for x in wiki_data]
y = [x[1][0] for x in wiki_data]
print(X)
print(y)
model.fit(X, y, batch_size=128, nb_epoch=100)
print('result')
pred = model.predict_classes(X, batch_size=128, verbose=1)

print(pred)
