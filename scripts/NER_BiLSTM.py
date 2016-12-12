# -*- coding: utf-8 -*-
import numpy as np
from keras.layers.advanced_activations import SReLU
from keras import callbacks as ckbs
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
from sklearn.model_selection import train_test_split

class FullModelCheckpoint(ckbs.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    pkl.dump([self.model.to_json(), self.model.get_weights()], open(filepath, 'wb'))
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            pkl.dump([self.model.to_json(), self.model.get_weights()], open(filepath, 'wb'))

def load_neural_network(file_from):
    (nn_arch, nn_weights_path) = pickle.load(open(file_from, 'rb'))
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



#####################################
#
# Create the  Network
#
#####################################

n_out = len(label2Idx)
  
model = Sequential()
model.add(Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False))

model.add(LSTM(100, return_sequences=True, dropout_W=0.2))
model.add(LSTM(100, return_sequences=True, dropout_W=0.2))
model.add(LSTM(100, dropout_W=0.2))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(SReLU())
model.add(Dense(n_out, activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-7, momentum=0.0, nesterov=False, clipvalue=3) 
rmsprop = RMSprop(clipvalue=3) 
nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='sparse_categorical_crossentropy', optimizer=nadam,metrics=['accuracy', 'precision', 'recall', 'categorical_accuracy', 'fmeasure'])

model.summary()
checkpointer = FullModelCheckpoint(filepath= "../models/best_train_model", verbose=1, save_best_only=True)

##################################
#
# Training of the Network
#
#################################

print( "%d test sentences" % len(dev_data))
dev = np.array(dev_data)
print(dev.shape)
X = [x[0][0] for x in dev_data]
y = [x[1][0] for x in dev_data]
#print(X)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model.fit(X_train, y_train, batch_size=128, nb_epoch=100, validation_data=(X_test, y_test), callbacks=[checkpointer])
print('result')
#pred = model.predict_classes(X_test, batch_size=128, verbose=1)

#print(pred)
