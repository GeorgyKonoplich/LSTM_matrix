import numpy as np
import cPickle as pkl
import gzip
import theano
from gensim.models.word2vec import Word2Vec
from itertools import islice
import pandas as pd
import math

#embeddingsPath = './all.norm-sz100-w10-cb0-it1-min100.w2v'#'./ruwikiruscorpora.model.bin.gz'

#wv = Word2Vec.load_word2vec_format(embeddingsPath, binary=True, unicode_errors='ignore')
#wv.init_sims(replace=True)
#from itertools import islice
#print list(islice(wv.vocab, 130, 132))

folder = '../data/processed/Dialog_test/'
feature_file = folder + 'NER_testset.features.csv'
target_file = folder + 'NER_testset.targets.csv'

feature_data = pd.read_csv(feature_file, sep = ';')
target_data = pd.read_csv(target_file)

words = ['', '2dom_', '1dom_', '-5_' , '-5_2dom_', '-5_1dom_', '-4_', '-4_2dom_', '-4_1dom_', '-3_',
         '-3_2dom_', '-3_1dom_', '-2_', '-2_2dom_', '-2_1dom_', '-1_', '-1_2dom_', '-1_1dom_', '+5_', '+5_2dom_', '+5_1dom_',
         '+4_', '+4_2dom_', '+4_1dom_', '+3_', '+3_2dom_', '+3_1dom_', '+2_', '+2_2dom_', '+2_1dom_', '+1_', '+1_2dom_', '+1_1dom_']

features = ['prefix4', 'prefix3', 'prefix2', 'prefix1', 'postfix4', 'postfix3', 'postfix2', 'postfix1', 'posStart', 'pos',
            'link', 'len', 'lemma', 'isupper', 'istitle', 'islower', 'isdigit', 'isalpha', 'isalnum', 'isLastWord', 'isFirstWord', 'grm', 'forma', 'dom', 'ID']


def createDataset(features_data, targets_data, label2Idx, case2Idx):

    wordEmbeddings = []
    wordIndices = []
    caseIndices = []
    labelIndices = []
    dataset = []
    for i in range(0, len(targets_data)):
        row = features_data.iloc[i]
        for word in words:
            print(word+'forma')
            wordforma = row[word+'forma']
            print(wordforma)
            v = []
            for ft in features:
                print(type(row[word+ft]))
                #if math.isnan(row[word+ft]):
                #    v.append(0)
                #else:
                v.append(row[word+ft])

            wordEmbeddings.append(v)
            wordIndices.append(len(wordEmbeddings) - 1)
            if type(wordforma) != 'str':
                caseIndices.append(case2Idx['other'])
            else:
                caseIndices.append(getCasing(wordforma, case2Idx))
        labelIndices.append(label2Idx[targets_data.iloc[i]['mark']])
        #break
        #dataset.append([wordIndices, caseIndices, labelIndices])
        dataset.append([wordIndices, labelIndices])
    wordEmbeddings = np.array(wordEmbeddings)


    return dataset, wordEmbeddings

def getCasing(word, caseLookup):   
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
   
    return caseLookup[casing]
           
outputFilePath = folder + 'pkl/data.pkl.gz'
embeddingsPklPath = folder + 'pkl/embeddings.pkl.gz'

labelSet = set()

for i in range(0, len(target_data)):
    labelSet.add(target_data.iloc[i]['mark'])


label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)


case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}
caseEmbeddings = np.identity(len(case2Idx), dtype=theano.config.floatX)

dataset, wordEmbeddings = createDataset(feature_data, target_data, label2Idx, case2Idx)

embeddings = {'wordEmbeddings': wordEmbeddings, 'caseEmbeddings': caseEmbeddings, 'label2Idx': label2Idx}

f = gzip.open(embeddingsPklPath, 'wb')
pkl.dump(embeddings, f, -1)
f.close()


f = gzip.open(outputFilePath, 'wb')
pkl.dump(dataset, f, -1)
f.close()
