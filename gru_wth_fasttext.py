#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:51:53 2021

@author: kaan
"""

#Needs to be tuned more
#Consider layer normalization in case num_layers>1

from gensim.models.fasttext import FastText
import nltk
import gc
import csv
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.model_selection import train_test_split        # train-test split'i bununla yapıyoruz
from sklearn.feature_extraction.text import TfidfVectorizer # TFIDF kullanmak için 
from sklearn.preprocessing import StandardScaler            # to scale input data

# for metrics and evaluation of results
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score,  f1_score, classification_report
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder

from torch import nn, optim, topk
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch
import torchtext
from torchtext.data import get_tokenizer
import scipy
import torch.nn.functional as F

import io
tokenizer = get_tokenizer("basic_english")


#%% parameters

'''                              
#Optimizer parameters           93% ttc3600         or even 94
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE =0.5
MOMENTUM = 0.9
DROPOUT = 0.1

#LSTM parameters
VECTOR_SIZE = 300      #Input size and the vector size are the same
INPUT_SIZE = 300

HIDDEN_SIZE = 300

NUM_LAYERS = 1

SENTENCE_LENGTH = 200  # max text length is 3125

SHUFFLE = True
'''
#Optimizer parameters
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE =0.5
MOMENTUM = 0.9
DROPOUT = 0.1

#LSTM parameters
VECTOR_SIZE = 300      #Input size and the vector size are the same
INPUT_SIZE = 300

HIDDEN_SIZE = 300

NUM_LAYERS = 1

SENTENCE_LENGTH = 200  # max text length is 3125

SHUFFLE = True

#%%obtain vectors

print('---------------------------------------------------------------------')
print('reading FastText training vectors ...')


fname ='/home/kaan/Desktop/cc.tr.300.vec'


fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
n, d = map(int, fin.readline().split())
vectors = {}
for line in fin:
    tokens = line.rstrip().split(' ')
    vectors[tokens[0]] = [float(i) for i in tokens[1:]]
# vector is a dictionary with keys corresponding to each vectorized word (with length 300)
# vector[','] as an example ( to reach the vectors)


#%%obtain data
file_name = '/home/kaan/Downloads/1-ttc3600.xlsx'
page = 'cleaned'

#file_name = '/home/kaan/Downloads/2-ttc4900.xlsx'
#page = 'sw_lem'

language =         'turkish'
scoring =          'accuracy'    
useStopwords =     False 
useLemmatize =     False

# %% import data
data = pd.read_excel(file_name, page)            #, encoding = 'utf-8')

# summarize dataset
print("input data shape:   ", data.shape)
print(data.groupby('class').size())
print(data['class'].nunique())

NUM_LABEL = data['class'].nunique()


le = LabelEncoder().fit(data["class"])
data['encodedcat'] = le.transform(data['class'])

train_news, test_news, train_topics, test_topics = train_test_split(data['description'],data['encodedcat'],test_size=.2,stratify=data['class'],random_state=42)

train_news=train_news.tolist()
test_news = test_news.tolist()

training_tokens=[]                                        #training tokens

for i,row in enumerate(train_news):
    training_tokens.append(tokenizer(row))

testing_tokens = []                                       #testing tokens

for i,row in enumerate(test_news):
    testing_tokens.append(tokenizer(row))

y_train = torch.tensor(train_topics.values)                 #converts the panda object to a tensor 
y_test = torch.tensor(test_topics.values)

#%%
torch.cuda.empty_cache()


#%% Prepare training dataset
#train dataset for dataloader


training_tuples = []

for j,_  in enumerate(train_news):
    a = torch.zeros( SENTENCE_LENGTH,VECTOR_SIZE)
    for num , i in enumerate(training_tokens[j]):   
            
        if num < SENTENCE_LENGTH:
        
            try:
                    a[num] = torch.Tensor(vectors[i]).reshape(1,VECTOR_SIZE)  # construct  matrices for each sentence
                    
            except KeyError:
                    continue
            
        else: continue
    a = StandardScaler(with_mean = 0).fit_transform(a)
    #a = nn.functional.normalize(a)
    tuple1=(a,y_train[j])

    training_tuples.append(tuple1)                              # construct tuples with data and labels


#%% Prepare test dataset
#test dataset for test dataloader
testing_tuples = []

# buraya j = 0 yazarsak çalışmıyor?? 

for j,_  in enumerate(test_news):
    b = torch.zeros( SENTENCE_LENGTH,VECTOR_SIZE)
    for num2 , i2 in enumerate(testing_tokens[j]):   
        if num2 < SENTENCE_LENGTH:
        
            try:
                    b[num2] = torch.Tensor(vectors[i2]).reshape(1,VECTOR_SIZE)  # construct  matrices for each sentence
                    
            except KeyError:
                    continue
            
        else: continue    
    b = StandardScaler(with_mean = 0).fit_transform(b)
    #b = nn.functional.normalize(b)             
    tuple2=(b,y_test[j])
    
    testing_tuples.append(tuple2)                              # construct tuples with data and labels

#%%
'''
print(testing_tokens[0])
vec, label  = testing_tuples[0]
print(vec.shape)
print(vec[1,:])
print(vec[1,:].shape)
print(vectors['5'])
'''
#%% Cuda

if torch.cuda.is_available():
    device = torch.device("cuda:2")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print('Device name:', torch.cuda.get_device_name(0))
else: 
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

torch.cuda.empty_cache()


#device = torch.device("cpu")

#%%

NUM_LABEL = data['class'].nunique()
TEST_BATCH = test_topics.__len__()

train_dl= DataLoader(training_tuples,batch_size=BATCH_SIZE,shuffle=SHUFFLE,pin_memory=True)
test_dl = DataLoader(testing_tuples, batch_size =TEST_BATCH, shuffle = SHUFFLE, pin_memory = True)


#%%Model design

print("---Model Design------------------------------")
class RNNet(nn.Module):
    def __init__(self,input_layer, hidden_size,num_layers):
        super(RNNet,self).__init__()
        self.input_layer = input_layer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.GRU = nn.GRU(input_size = input_layer,hidden_size = hidden_size, num_layers = num_layers)
        self.linear = nn.Linear(hidden_size,NUM_LABEL).to('cuda:1')
        
        
    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.shape[1], self.hidden_size, device = device)
        
        
        
        output, hn = self.GRU(x,h0) 
        
        output= output[-1,:,:]

        output = self.linear(output)
        output = F.log_softmax(output, dim = 0)
        return output   


model = RNNet (INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)

# Main function

train_losses = []
test_losses = []
#test_accuracies = []
#val_outputs= []
f1_score_list = []

criterion =  nn.CrossEntropyLoss()

def train_model(model,train_dl,test_dl,epochs):
       
    #gc.collect()
    
    optimizer= optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM)
    #optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    for epoch in range(epochs):
        
        
        for train_data, train_label_data in train_dl: 
            model.train()
            train_data = train_data.to(device).float()
            train_data = train_data.transpose(0,1)
            #train_data =train_data.permute(1,0,2)
            train_label_data = train_label_data.to(device)
            output = model(train_data)

            loss = criterion(output,train_label_data)
            loss.backward()
            train_loss = loss.item()
            
            #train_losses.append(train_loss)
            optimizer.step()
            optimizer.zero_grad()
            loss.detach()
            
            validation_f1_score,val_loss = evaluate_model(model,test_dl)
            
            f1_score_list.append(validation_f1_score)
            #test_losses.append(val_loss)
            torch.cuda.empty_cache()
            print(f"Epoch: {epoch+1}/{epochs}..", f"Training loss: {train_loss:.3f}", f"Validation loss: {val_loss:.3f} " , f"Validation F1 Score: {validation_f1_score:.3f}")
    
    plt.figure(figsize=(12,5))
    plt.title(f"Batch size / Learning rate = {BATCH_SIZE, LEARNING_RATE}")
    plt.subplot(121)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(train_losses,label='Training loss')
    plt.plot(test_losses, label ='Validation loss')
    plt.legend(frameon=True);
    plt.subplot(122)
    plt.xlabel('epochs')
    plt.ylabel('F1 score')
    plt.plot(f1_score_list,label='F1 score')
        


def evaluate_model(model,test_dl):
    model.eval()
    #actual = []
    for test_data,test_label_data in test_dl:
        test_data = test_data.to(device).float()
        test_data = test_data.transpose(0,1)
        #test_data =test_data.permute(1,0,2)
        test_label_data = test_label_data.to(device)
        
        val_out = model(test_data)
        
        val_loss= criterion(val_out,test_label_data)
        
        val_loss.detach()
        #targets=test_label_data.cpu().numpy()
                    
        _, predictions = torch.max(val_out,dim=1) 
        #accuracy = accuracy_score(predictions, targets)
        #test_accuracies.append(accuracy)
        #actual.append(predictions) #Creates a list with all the outputs
        test_label_data = test_label_data.cpu()
        predictions = predictions.cpu()
        f1_score_result=f1_score(test_label_data,predictions,average='weighted')     
        torch.cuda.empty_cache()
    return f1_score_result,val_loss
    #print(f"Validation loss: {val_loss:.3f}" ,f"Validation accuracy:{accuracy:.3f}")
     
    


print("Training....")
train_model(model,train_dl,test_dl, EPOCHS)