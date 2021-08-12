#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 15:39:10 2021

@author: kaan
"""


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

import io

import torch
import torchtext
import scipy
import torch.nn.functional as F
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer("basic_english")


# my files
#import news_preprocess

#%% read  
# file_name = '/home/kaan/Downloads/1-ttc3600.xlsx'
# page = 'cleaned'
file_name = 'data/1-ttc3600.xlsx'
page = 'dnm'

#file_name = '/home/kaan/Downloads/2-ttc4900.xlsx'
#page = 'sw_lem'

language =         'turkish'
scoring =          'accuracy'    
useStopwords =     False 
useLemmatize =     False

#%% parameters
'''
NO_EPOCHS = 8  # Tuned parameters for ttc3600 cleaned
LEARNING_RATE = 0.0003
MOMENTUM = 0.9
KERNEL_SIZE = 10
KERNEL_SIZE1 = 10
KERNEL_SIZE2 = 3
POOLING = 3
DROPOUT = 0.15
STRIDE = 1
BATCH_SIZE = 80 # or 64
'''
'''
NO_EPOCHS = 10 # Tuned parameters for ttc4900 sw_lem
LEARNING_RATE = 0.00005
MOMENTUM = 0.9
KERNEL_SIZE = 17
KERNEL_SIZE1 = 10
KERNEL_SIZE2 = 3
POOLING = 2
DROPOUT = 0.1
STRIDE = 1
BATCH_SIZE = 32
'''
NO_EPOCHS     = 5             # Tuned parameters for ttc3600 cleaned
LEARNING_RATE = 0.01
MOMENTUM      = 0.9
KERNEL_SIZE   = 10

POOLING       = 4
DROPOUT       = 0.1
STRIDE        = 1
BATCH_SIZE    = 64        # or 64


#LSTM parameters
VECTOR_SIZE = 300      #Input size and the vector size are the same
INPUT_SIZE = 300

HIDDEN_SIZE = 100

NUM_LAYERS = 1

SENTENCE_LENGTH = 400  # max text length is 3125

SHUFFLE = True


# random state
RS = 42
torch.manual_seed(RS)
torch.cuda.manual_seed(RS)
np.random.seed(RS)

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
                  
        tuple2=(b,y_test[j])
        print(b.shape)
        testing_tuples.append(tuple2)                              # construct tuples with data and labels


#%%

NUM_LABEL = data['class'].nunique()
TEST_BATCH = test_topics.__len__()

train_dl= DataLoader(training_tuples,batch_size=BATCH_SIZE,shuffle=SHUFFLE,pin_memory=True)
test_dl = DataLoader(testing_tuples, batch_size =BATCH_SIZE, shuffle = SHUFFLE, pin_memory = True)

#%%
for a , b in train_dl :
    print(a.shape)
#%% Cuda

if torch.cuda.is_available():
    device = torch.device("cuda:2")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print('Device name:', torch.cuda.get_device_name(0))
else: 
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

torch.cuda.empty_cache()

#%%
#https://www.machinecurve.com/index.php/2021/03/29/batch-normalization-with-pytorch/

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.LazyConv1d(VECTOR_SIZE, KERNEL_SIZE,STRIDE)
        self.pool1=nn.MaxPool1d(POOLING)
        self.hidden = nn.LazyLinear(32)
        self.norm1 = nn.BatchNorm1d(32)
        self.hidden2 = nn.Linear(32,NUM_LABEL)
        
    def forward(self,x):
        x=self.conv1(x)    
        x=F.relu(x)
        x=self.pool1(x)
        x=x.view(x.size(0),-1) 
        x=self.norm1(self.hidden(x))
        x= self.hidden2(x)     
        output=F.softmax(x,dim=1)
        return output   


#https://www.youtube.com/watch?v=8YsZXTpFRO0
#https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412/2
        
model=Net()


#%%
train_losses=[]
test_losses=[]
test_accuracies=[]
val_outputs=[]
f1_score_list = []
criterion = nn.CrossEntropyLoss()
#optimizer= optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM)


def train_model(model,train_dl,epochs):
    model.train()
    optimizer= optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM)
    for epoch in range(epochs):
        
        for train_data, train_label_data in train_dl: 
           
            train_data=train_data.permute(0,2,1)   #[batch_size, in_channels, len]
            
            output = model.forward(train_data)
            loss = criterion(output,train_label_data)
            
            loss.backward()
            train_loss = loss.item()
            train_losses.append(train_loss)
            optimizer.step()
            optimizer.zero_grad()
            validation_f1_score,val_loss = evaluate_model(model,test_dl)
            f1_score_list.append(validation_f1_score)
            test_losses.append(val_loss)
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
    actual = []
    with torch.no_grad():
        for test_data,test_label_data in test_dl:
            test_data=test_data.permute(0,2,1) 
            val_out = model.forward(test_data)
            val_outputs.append(val_out)
            val_loss= criterion(val_out,test_label_data)
            
            targets=test_label_data.numpy()
                        
            _, predictions = torch.max(val_out,dim=1) 
            accuracy = accuracy_score(predictions, targets)
            test_accuracies.append(accuracy)
            actual.append(predictions) #Creates a list with all the outputs
            f1_score_result=f1_score(test_label_data,predictions,average='weighted')     
            #print("Validation loss :f"F1 score result: {f1_score_result:.3f}")            
            return f1_score_result,val_loss
                
train_model(model,train_dl,NO_EPOCHS) 
#evaluate_model(model,test_dl)  