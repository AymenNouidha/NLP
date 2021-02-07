import torch
from torch import nn
from torchcrf import CRF
from numpy import array
from random import shuffle
import pandas as pd
import spacy
import bcolz
import pickle
import numpy as np
import xml.etree.ElementTree as et
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
import xml.etree.ElementTree as et
import itertools
import random

glove_path = './glove'

def replace(file_path):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                linef = line.replace("&", "and")
                new_file.write(linef.replace(linef, '<ref>'+linef+'</ref>\n'))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    #remove(file_path)
    #Move new file
    move(abs_path, file_path)
    line_prepender(file_path)

def line_prepender(filename):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('<?xml version="1.0"?>\n<data>\n' + content)
    with open(filename, 'a') as f:
        f.write('</data>')
#replace(file_path)

def printing(file_path):

    spacy_en = spacy.load('en')
    xmlTree = et.parse(file_path)
    xroot = xmlTree.getroot()
    tags = [elem.tag for elem in xroot.iter()]
    tags = tags[2:]
    tags = [list(y) for x, y in itertools.groupby(tags, lambda z: z == 'ref') if not x]
    for i, j in enumerate(tags):
        print(j)
    tagsE = [elem.text for elem in xroot.iter()]
    tagsE = tagsE[2:]
    tagsE = [list(y) for x, y in itertools.groupby(tagsE, lambda z: z == None) if not x]
    tagsE = [[elem.strip() for elem in sublist] for sublist in tagsE]
    print(tagsE)
    zipped = list(zip(tags, tagsE))
    random.shuffle(zipped)
    tags, elems = zip(*zipped)
    tags = list(tags)
    elems = list(elems)
    #List Comprehension
    #carachtersList = [[list(word) for word in sublist]for sublist in list(elems)]
    #(tok.is_punct | tok.is_space)
    BIElem = [[[tok.text for tok in spacy_en.tokenizer(word) if tok.text not in [',', ' ']] for word in sublist] for sublist in list(elems)]
    carachtersList = [[[list(word)  for word in field] for field in line] for line in BIElem ]
    caractersSet = list(set([c for line in carachtersList for field in line for word in field for c in word]))
    di = getHeader(tags)
    return (tags, elems, BIElem, carachtersList, caractersSet, di)

def getHeader(elems):
    header = list(set(list(itertools.chain.from_iterable(elems))))
    return header

def Convert(lst):
    res_dct = {lst[i]: i+1 for i in range(len(lst))}
    return res_dct

def prepare_batch(train, tags, dic, batch_size=32):
    number_batches = (len(train) + (batch_size-1))//batch_size
    last_batch = len(train)%batch_size
    if(last_batch==0):
        last_batch = batch_size
    lines = []
    tagsF = []
    inter2 = []
    last = batch_size
    for i in range(number_batches):
        if(i==number_batches-1):
            last = last_batch
        tagsF = []
        lines = []
        for j in range(i*batch_size, i*batch_size + last): #line
            for k in range(len(train[j])): #field
                    for count, item in enumerate(train[j][k]): #BI Subfield
                        #for l in range(len(item)):
                        if count == 0:
                            inter2.append(dic.index(tags[j][k])*2)
                        else:
                            inter2.append(dic.index(tags[j][k])*2 + 1)
                    #tagsF.append(inter2)
                    #inter2 = []
            tagsF.append(inter2)
            lines.append(train[j])
            inter2 = []
        lines = [[word for field in line for word in field] for line in lines ]
        zipped = zip(lines, tagsF, tags[i*batch_size:i*batch_size + last], train[i*batch_size:i*batch_size + last])
        zipped = sorted(zipped, key = lambda t: len(t[0]), reverse=True)
        lines, tagsF, Otags, Olines = zip(*zipped)
        length = [len(x) for x in lines]
        yield lines, tagsF, length, Otags, Olines

def preprareEmb():
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

    with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

def createCarachtersDic(target_vocab):
    charToindex = {}
    indexToChar = {}
    charToindex['unk'] = 0
    charToindex['pad'] = 1
    indexToChar[1] = 'pad'
    indexToChar[0] = 'unk'
    count = 2
    for i, c in enumerate(target_vocab):
        charToindex[c] = count
        indexToChar[count] = c
        count = count + 1
    return (charToindex, indexToChar)

def citos(index, vocabDict):
    return vocabDict[index]

def buildWeightMatrix(target_vocab):
    preprareEmb()
    vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    target = list(set([word for entry in target_vocab for field in entry for word in field ]))
    matrix_len = len(target)
    vocabDict = {}
    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0
    vocabDict['unk'] = 0
    weights_matrix[0] = glove['unk']
    vocabDict['pad'] = 1
    weights_matrix[1] = glove['pad']
    count = 2
    for i, word in enumerate(target):
        try: 
            weights_matrix[count] = glove[word]
            vocabDict[word] = count
            count += 1
        except KeyError:
            pass
    return (weights_matrix, vocabDict, glove)

def stoi(word, vocabDict):
    if word in vocabDict:
        return vocabDict[word]
    return 0

def create_emb_layer(weights_matrix, non_trainable=True):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

def pad(batch, tags, lengths, dicti):
    retBatch = []
    retTags = []
    temp = []
    tagTemp = []
    for count, line in enumerate(batch):
        retBatch.append(line + ['pad']*(lengths[0] - lengths[count]))
        temp = [[stoi(w, dicti) for w in s]for s in retBatch]
    for count, line in enumerate(tags):
        retTags.append(line + [1]*(lengths[0] - lengths[count]))
        tagtemp = [[w for w in s]for s in retTags]
    return torch.tensor(temp, device=0), torch.tensor(tagtemp, device=0)

num_tags = 34

def split(dataset):
    return(dataset[0:350], dataset[350:425], dataset[425:])

def initializeModel(model):
    for name, param in model.named_parameters():
        print(name)
        #if 'weight' in name:
            #nn.init.normal_(param.data, std=0.01)
        #else:
            #nn.init.constant_(param.data, 0)