# -*- coding: utf-8 -*-
import numpy as np

import random


import time
import gzip
import pickle as pkl


import BIOF1Validation

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
#tokens = Sequential()

#tokens.add(Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False))
#casing = Sequential()
#casing.add(Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)) 
  
model = Sequential()
model.add(Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False))
#model.add(Merge([tokens, casing], mode='concat'))  
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
def iterate_minibatches(dataset, startIdx, endIdx): 
    endIdx = min(len(dataset), endIdx)
    
    for idx in range(startIdx, endIdx):
        tokens, labels = dataset[idx]        
        #print(labels)
        #print(len(labels))    
        #labels = np.expand_dims([labels], -1)     
        yield np.asarray([labels]), np.asarray([tokens])


def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    for tokens, labels in dataset:    
        tokens = np.asarray([tokens])     
        pred = model.predict_classes(tokens, verbose=False)[0]               
        correctLabels.append(labels)
        predLabels.append(pred)
        
        
    return predLabels, correctLabels
        
number_of_epochs = 20
stepsize = 100
#print "%d epochs" % number_of_epochs

#print "%d train sentences" % len(train_data)
#print "%d dev sentences" % len(dev_data)
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
'''
for epoch in range(number_of_epochs):    
    print( "--------- Epoch %d -----------" % epoch)
    random.shuffle(wiki_data)
    for startIdx in range(0, len(wiki_data), stepsize):
        start_time = time.time()    
        for batch in iterate_minibatches(wiki_data, startIdx, startIdx+stepsize):
            
            labels, tokens = batch       
            #print(labels.shape)
            #print(tokens.shape)
            model.train_on_batch(tokens, labels)   
        print( "%.2f sec for training" % (time.time() - start_time))
        
        
        
        #Test Dataset       
        predLabels, correctLabels = tag_dataset(wiki_data)
        print(predLabels)
        print(correctLabels)
        #pre_test, rec_test, f1_test= BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label)
        #print "Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test)
        
        print("finish")
'''
