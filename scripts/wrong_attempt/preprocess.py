import numpy as np
import pickle as pkl
import gzip
import theano
from gensim.models.word2vec import Word2Vec
from itertools import islice
import pandas as pd
import math

folder = '../data/processed/'
files = ['Dialog_test/NER_testset', 'Dialog_train/NER_devset', 'Wiki_set/WikiNER_part1']
feature_file = '.features.csv'
target_file = '.targets.csv'

words = ['', '2dom_', '1dom_', '-5_' , '-5_2dom_', '-5_1dom_', '-4_', '-4_2dom_', '-4_1dom_', '-3_',
         '-3_2dom_', '-3_1dom_', '-2_', '-2_2dom_', '-2_1dom_', '-1_', '-1_2dom_', '-1_1dom_', '+5_', '+5_2dom_', '+5_1dom_',
         '+4_', '+4_2dom_', '+4_1dom_', '+3_', '+3_2dom_', '+3_1dom_', '+2_', '+2_2dom_', '+2_1dom_', '+1_', '+1_2dom_', '+1_1dom_']

features = ['prefix4', 'prefix3', 'prefix2', 'prefix1', 'postfix4', 'postfix3', 'postfix2', 'postfix1', 'posStart', 'pos',
            'link', 'len', 'lemma', 'isupper', 'istitle', 'islower', 'isdigit', 'isalpha', 'isalnum', 'isLastWord', 'isFirstWord', 'grm', 'forma', 'dom', 'ID']

str_features = ['forma', 'lemma', 'prefix3', 'prefix2', 'prefix1', 'pos', 'prefix4', 'postfix4', 'postfix1', 'postfix3', 'postfix2', 'link', 'grm']

outputFilePath = folder + 'pkl/data.pkl.gz'
embeddingsPklPath = folder + 'pkl/embeddings.pkl.gz'


def createDataset(label2Idx):
    code_table = {}
    for ft in str_features:
        code_table[ft] = {}
    wordEmbeddings = []
    fl = gzip.open(outputFilePath, 'wb')
    for f in files:
        feature_data = pd.read_csv(folder + f + feature_file, sep=';')
        target_data = pd.read_csv(folder + f + target_file)
        dataset = []
        for i in range(0, len(feature_data)):
            wordIndices = []
            labelIndices = []
            row = feature_data.iloc[i]
            for word in words:
                #print(word+'forma')
                wordforma = row[word+'forma']
                #print(wordforma)
                v = []
                for ft in features:
                    x = row[word+ft]
                    if ft in str_features:
                        if row[word+ft] in code_table[ft]:
                            x = code_table[ft][row[word+ft]]
                        else:
                            code_table[ft][row[word + ft]] = len(code_table[ft])
                            x = code_table[ft][row[word + ft]]
                    #print(type(row[word+ft]))
                    #if math.isnan(row[word+ft]):
                    #    v.append(0)
                    #else:
                    v.append(x)

                wordEmbeddings.append(v)
                wordIndices.append(len(wordEmbeddings) - 1)

            labelIndices.append(label2Idx[target_data.iloc[i]['mark']])
            dataset.append([wordIndices, labelIndices])
        
        print('create_data')
        pkl.dump(dataset, fl, -1)

        #break

    wordEmbeddings = np.array(wordEmbeddings)
    fl.close()
    return wordEmbeddings, code_table

def get_string_column():
    featureSet = set()

    for f in files:
        feature_data = pd.read_csv(folder + f + feature_file, sep=';')
        for i in range(0, len(feature_data)):
            row = feature_data.iloc[i]
            for ft in features:
                if type(row[ft]) == 'str':
                    featureSet.add(ft)
    print(featureSet)
    return  featureSet


def get_labels():
    labelSet = set()

    for f in files:
        target_data = pd.read_csv(folder + f + target_file)
        for i in range(0, len(target_data)):
            labelSet.add(target_data.iloc[i]['mark'])

    return  labelSet


###################################################
#data preprocessing start

#featureSet = get_string_column()

labelSet = get_labels()
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)



wordEmbeddings, code_table = createDataset(label2Idx)

embeddings = {'wordEmbeddings': wordEmbeddings, 'label2Idx': label2Idx}

f = gzip.open(embeddingsPklPath, 'wb')
pkl.dump(embeddings, f, -1)
f.close()
