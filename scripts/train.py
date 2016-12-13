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
from sklearn.preprocessing import OneHotEncoder

f = gzip.open('../data/processed/pkl/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

label2Idx = embeddings['label2Idx']

#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}

f = gzip.open('../data/processed/pkl/data.pkl.gz', 'rb')
test_data = pkl.load(f)
dev_data = pkl.load(f)
#wiki_data = pkl.load(f)
f.close()

n_out = len(label2Idx)

def load_neural_network(file_from):
    (nn_arch, nn_weights_path) = pkl.load(open(file_from, 'rb'))
    nn = model_from_json(nn_arch)
    nn.set_weights(nn_weights_path)
    return nn

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

def get_model(matrix):
    emb_layers = []
    for param_num in range(matrix.shape[2]):
        model = Sequential()
        embedding_max_index = np.max(matrix[:, :, param_num]) + 2
        out_size = 3
        if embedding_max_index > 300:
            out_size = 300
        elif embedding_max_index > 100:
            out_size = 10
        model.add(Embedding(embedding_max_index, out_size, input_length=matrix.shape[1]))
        emb_layers.append(model)
    
    merged = Merge(emb_layers, mode = 'concat')
    return merged

def get_data(data):
    print( "%d test size" % len(data))
    data = np.array(data)
    print(data.shape)
    X = [x[0] for x in data]
    y = [x[1] for x in data]
    return np.array(X), np.array(y)


def matrix_train_val_test_split(matrix, target, number_of_samples_to_use=None, test_size=0.2, random_state=23):
   
    train_samples = int(matrix.shape[0]*1)
    X_train = matrix[:train_samples]
    X_test = matrix[train_samples:]
        
    y_train = target[:train_samples]
    y_test = target[train_samples:]
        
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    
    
    return  [X_train[:,:,i] for i in range(X_train.shape[2])], \
            [X_val[:,:,i] for i in range(X_val.shape[2])], \
            y_train, y_val

X,y = get_data(dev_data)
merged = get_model(X)
model = Sequential()
model.add(merged)
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
model.compile(loss='categorical_crossentropy', optimizer=nadam,metrics=['accuracy', 'precision', 'recall', 'categorical_accuracy', 'fmeasure'])

model.summary()
checkpointer = FullModelCheckpoint(filepath= "../models/full_best_train_model", verbose=1, save_best_only=True)

X_train, X_val, y_train, y_val = matrix_train_val_test_split(X, y)
print("data prepared")
y_train_dum = OneHotEncoder(sparse=False).fit_transform(y_train.reshape(-1, 1))
y_val_dum = OneHotEncoder(sparse=False).fit_transform(y_val.reshape(-1, 1))
model.fit(X_train, y_train_dum, batch_size=128, nb_epoch=100, validation_data=(X_val, y_val_dum), callbacks=[checkpointer])
