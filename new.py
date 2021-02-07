import torch
import torch.nn as nn
import spacy
import time
import pickle
from operator import truediv
from functools import partial
import numpy as np
from torchtext.data import Field, NestedField, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import random
from numpy import array
from joblib import Memory
import time
import os
from random import shuffle
from torchtext.vocab import Vectors
from utils import prepare_batch, pad, initializeModel, printing, preprareEmb, buildWeightMatrix, createCarachtersDic, stoi, citos, split
from sklearn.utils import shuffle
from model import BilstmCrf

modelPath = './model'
optimizerPath = './optimizer'
file_path = './tagged_references.txt'
database = './bases'

min_dev_loss = float('inf')

def Train(min_dev_loss, modelPath, optimizerPath):
    decay = 0
    patience = 0
    device = torch.device('cuda')
    
    #from torchtext.vocab import GloVe
    #fine_trained_vectors = GloVe(name='6B', dim=50)

    tags, elems, BIElem, carachtersList, caractersSet, header = printing(file_path)
    print('*//*/*/*/*/**//*/**/*/*/')
    print(elems[:2])
    print(BIElem[:2])
    print(carachtersList[0:2])
    print(caractersSet)
    print(len(caractersSet))
    print(kj)
    BIElemT, BIElemV, BIElemTE = split(BIElem)
    TagsT, TagsV, TagsTE = split(tags)
    weight, dicts, glove = buildWeightMatrix(BIElem)
    charToindex, indexToChar = createCarachtersDic(caractersSet)
    pickle.dump(TagsTE, open(f'{database}/testTags.pkl', 'wb'))
    pickle.dump(BIElemTE, open(f'{database}/test.pkl', 'wb'))
    pickle.dump(dicts, open(f'{database}/dicti.pkl', 'wb'))
    pickle.dump(header, open(f'{database}/header.pkl', 'wb'))
    

    model = BilstmCrf(50, 50, weight).to('cuda:0')
    initializeModel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    train_iter = 0
    record_loss_sum, record_word_sum, record_batch_size = 0, 0, 0
    validation_loss_sum, validation_word_sum, validation_batch_size = 0, 0, 0
    record_start_time, validation_start_time = time.time(), time.time()
    
    print('Training...')
    for epoch in range(1):
        for lines, tags, length, _, _ in prepare_batch(BIElemT, TagsT, header, batch_size=3):
            train_iter += 1
            Batch, rettags = pad(lines, tags, length, dicts)
            #features = TEXT.vocab.vectors[Batch].reshape((Batch).size()[0], (Batch).size()[1], -1)
            optimizer.zero_grad()
            batch_loss = model(Batch, rettags, length)
            loss = batch_loss.item()
            print('*********')
            print(type(loss))
            print(loss)
            print('*********')
            batch_loss.backward()
            optimizer.step()
            record_loss_sum += loss
            record_batch_size += len(length)
            record_word_sum += sum(length)
            validation_loss_sum += loss
            validation_batch_size += len(length)
            validation_word_sum += sum(length)
            if train_iter % 5 == 0:
                print('log: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, record_word_sum / (time.time() - record_start_time),
                       record_loss_sum / record_batch_size, time.time() - record_start_time))
                record_loss_sum, record_batch_size, record_word_sum = 0, 0, 0
                record_start_time = time.time()
            if train_iter % 15 == 0:
                print('dev: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, validation_word_sum / (time.time() - validation_start_time),
                       validation_loss_sum / validation_batch_size, time.time() - validation_start_time))
                validation_loss_sum, validation_word_sum, validation_batch_size = 0, 0, 0
                dev_loss = dev_lossF(model, BIElemV, dicts, TagsV, header)
                if dev_loss < min_dev_loss * 0.95 :
                    min_dev_loss = dev_loss
                    torch.save(model, os.path.join(modelPath,'model.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(optimizerPath,'optimizer.pth'))
                    patience = 0
                else:
                    patience += 1
                    if patience == 20:
                        decay += 1
                        if decay == 10:
                            print('Early stop, model saved to %s'% modelPath)
                            return
                        lr = optimizer.param_group[0]['lr']*0.1
                        model = BilstmCrf.load(os.path.join(modelPath,'model.pth'), device)
                        optimizer.load_state_dict(torch.load(os.path.join(optimizerPath,'optimizer.pth')))
                        for param_group in optimizer.param_group:
                            param_group['lr'] = lr
                        patience = 0
                print('dev: epoch %d, iter %d, dev_loss %f, patience %d, decay_num %d' %
                      (epoch + 1, train_iter, dev_loss, patience, decay))
                validation_start_time = time.time()
        print('Reached %d epochs, Save result model to %s' % (epoch, modelPath))
        break

    #print(list(model.parameters()))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    #from functools import reduce
    #ki = reduce(lambda x,y: torch.cat((x,y)), li[:])
    #ki = tuple(li[0])
    #lik = torch.cat(ki, 0)
    #lik2 = torch.stack(ki, 0)
    return (BIElemTE, dicts, TagsTE, header)

def dev_lossF(model, df, dicti, TagsV, header):

    loss, n_sentences = 0, 0
    isTraining = model.training
    model.eval()
    with torch.no_grad():
        for lines, tags, length, _, _ in prepare_batch(df, TagsV, header):
            Batch, rettags = pad(lines, tags, length, dicti)
            batch_loss = model(Batch, rettags, length)
            loss += batch_loss.item()
            n_sentences += len(lines)
    model.train(isTraining)
    return loss/n_sentences

def Test(modelPath, optimizerPath, dfT, dicti, TagsTE, header):
     #model = BilstmCrf.load(os.path.join(modelPath,'model.pth'), torch.device('cuda'))
     model = torch.load(os.path.join(modelPath,'model.pth'), torch.device('cuda'))
     print('Start the Testing...')
     result = open('Results.txt', 'w')
     model.eval()
     sums = 0
     inplace = 0
     count = 0
     predl = [0]*13
     reall = [0]*13
     with torch.no_grad():
        for lines, tags, length, originalTags, oLines in prepare_batch(dfT, TagsTE, header, batch_size=1):
            Batch, rettags = pad(lines, tags, length, dicti)
            predTags = model.predict(Batch, length)
            for sent, tag, pred, otags, oline in zip(lines, tags, predTags, originalTags, oLines):
                j = 0
                prev = 0
                print('**********')
                print(oline)
                print(sent)
                print(otags)
                print(tag)
                print(pred)
                print('**********')
                for i in range(len(oline)):
                    j = j + len(oline[i])
                    for k in range(prev, j):
                        reall[header.index(otags[i])] += 1
                        if(tag[k]==pred[k]):
                            predl[header.index(otags[i])] += 1
                    prev = j
                for i in range(len(sent)):
                    sums = sums + 1
                    if(tag[i]==pred[i]):
                        inplace = inplace + 1
                print(str(sums))
                print(str(inplace))
                count = count + 1
        perc = (inplace/sums)*100
        print('Precision is: {:0.2f}'.format(perc))
        res = list(map(truediv, predl, reall))
        res = [el*100 for el in res]
        print(header)
        print(reall)
        print(res)
        print(predl)
   
dfT, dicti, TagsTE, header = Train(min_dev_loss, modelPath, optimizerPath)
TagsTE = pickle.load(open(f'{database}/testTags.pkl', 'rb'))
BIElemTE = pickle.load(open(f'{database}/test.pkl', 'rb'))
dicti = pickle.load(open(f'{database}/dicti.pkl', 'rb'))
header = pickle.load(open(f'{database}/header.pkl', 'rb'))
Test(modelPath, optimizerPath, BIElemTE, dicti, TagsTE, header)