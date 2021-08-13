#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:16:17 2021

@author: kaan
"""
#Model has to be changed 
#


#%% import libraries

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

# random state
RS = 42
torch.manual_seed(RS)
torch.cuda.manual_seed(RS)
np.random.seed(RS)

#%% read hphyperparameters 
file_name = '/home/kaan/Downloads/1-ttc3600.xlsx'
page = 'cleaned'

# file_name = '/home/kaan/Downloads/1-ttc4900.xlsx'
# page = 'sw_lem'

language =         'turkish'
scoring =          'accuracy'    
useStopwords =     False 
useLemmatize =     False

#%% parameyer
NO_EPOCHS = 15
LEARNING_RATE = 0.1
MOMENTUM = 0.9
BATCH_SIZE =720 #720
WEIGHT_DECAY = 0.01
DROPOUT1 = 0.5
DROPOUT2 =0.1
DROPOUT3 =0.7


VECTOR_SIZE = 300      #Input size and the vector size are the same
INPUT_SIZE = 300

HIDDEN_SIZE = 200


SENTENCE_LENGTH = 400  # max text length is 3125

SHUFFLE = True


#%%CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print('Device name:', torch.cuda.get_device_name(0))
else: 
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


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
    #a = StandardScaler(with_mean = 0).fit_transform(a)
    a = nn.functional.normalize(a)
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
    #b = StandardScaler(with_mean = 0).fit_transform(b)
    b = nn.functional.normalize(b)             
    tuple2=(b,y_test[j])
    
    testing_tuples.append(tuple2)                              # construct tuples with data and labels


#%%

NUM_LABEL = data['class'].nunique()
TEST_BATCH = test_topics.__len__()

train_dl= DataLoader(training_tuples,batch_size=BATCH_SIZE,shuffle=SHUFFLE,pin_memory=True)
test_dl = DataLoader(testing_tuples, batch_size =TEST_BATCH, shuffle = SHUFFLE, pin_memory = True)


#%% training function

train_losses = []
test_losses = []
test_accuracies = []
val_outputs= []
f1_score_list = []


#model = nn. Sequential(nn.Linear(x_train.shape[1],data['class'].nunique()),nn.ReLU(),nn.Dropout(0.5),nn.Softmax(dim=1))
criterion = nn.CrossEntropyLoss() 
          

'''
#4 layers
model = nn. Sequential(nn.Linear(x_train.shape[1],2048),
                       nn.BatchNorm1d(2048),
                       nn.ReLU(),
                       #nn.Dropout(DROPOUT1),
                       nn.Linear(2048,256),
                       nn.BatchNorm1d(256),
                       nn.ReLU(),
                       #nn.Dropout(0.5),
                       nn.Linear(256,64),
                       nn.BatchNorm1d(64),
                       nn.ReLU(),
                       #nn.Dropout(DROPOUT2),
                       nn.Linear(64,data['class'].nunique()),
                       nn.ReLU(),nn.Dropout(DROPOUT3),
                       nn.Softmax(dim=1))
criterion = nn.CrossEntropyLoss()            

'''
'''
#3 layers
model = nn. Sequential(nn.Linear(x_train.shape[1],4096),
                       #nn.BatchNorm1d(4096),
                       nn.ReLU(),
                       nn.Dropout(DROPOUT1),                    
                       nn.Linear(4096,64),
                       #nn.BatchNorm1d(64),
                       nn.ReLU(),
                       nn.Dropout(DROPOUT2),
                       nn.Linear(64,data['class'].nunique()),
                       nn.ReLU(),nn.Dropout(DROPOUT3),
                       nn.Softmax(dim=1))
criterion = nn.CrossEntropyLoss()


'''
'''

#2 layers 
model = nn. Sequential(nn.Linear(x_train.shape[1],4096),
                       nn.BatchNorm1d(4096),
                       nn.ReLU(),
                       nn.Dropout(DROPOUT1),                    

                       nn.Linear(4096,data['class'].nunique()),
                       nn.ReLU(),
                       nn.Dropout(DROPOUT3),
                       nn.Softmax(dim=1))
criterion = nn.CrossEntropyLoss()
'''


class LinNet(nn.Module):
    def __init__(self):
        super(LinNet,self).__init__()
        self.hidden1=nn.Linear(VECTOR_SIZE, 1)
        self.dropout1 = nn.Dropout(DROPOUT1)
        
        self.hidden2 = nn.Linear(SENTENCE_LENGTH, NUM_LABEL)
        self.dropout2 = nn.Dropout(DROPOUT2)
    def forward(self,x):
        x=self.dropout1(self.hidden1(x))    
        x=F.relu(x)
        #x=self.pool1(x)
        x=x.squeeze(2) 
        #x=self.hidden(x)
        x=self.dropout2(self.hidden2(x))    
        output=F.softmax(x,dim=1)
        return output 


model = LinNet()




def train_model(model,train_dl,epochs):
    model.train()
    
    optimizer= optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM)
    #optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    for epoch in range(epochs):
        
        for train_data, train_label_data in train_dl: 
            
            output = model(train_data)
            loss = criterion(output,train_label_data)
            loss.backward()
            train_loss = loss.item()
            train_losses.append(train_loss)
            optimizer.step()
        
            validation_f1_score,val_loss = evaluate_model(model,test_dl)
            f1_score_list.append(validation_f1_score)
            test_losses.append(val_loss)
            optimizer.zero_grad()
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
        
        
        
        
        
       # print(f"Epoch: {epoch+1}/{epochs}..", f"Training loss: {train_loss:.3f}")

def evaluate_model(model,test_dl):
    model.eval()
    actual = []
    for test_data,test_label_data in test_dl:
            
        val_out = model(test_data)
        val_outputs.append(val_out)
        #print(val_out.shape)
        val_loss= criterion(val_out,test_label_data)
        test_losses.append(val_loss)
        targets=test_label_data.numpy()
                    
        _, predictions = torch.max(val_out,dim=1) 
        #print(predictions)
        accuracy = accuracy_score(predictions, targets)
        test_accuracies.append(accuracy)
        actual.append(predictions) #Creates a list with all the outputs
        f1_score_result=f1_score(test_label_data,predictions,average='weighted')     
    return f1_score_result,val_loss
    #print(f"Validation loss: {val_loss:.3f}" ,f"Validation accuracy:{accuracy:.3f}")
     
    


print("Training....")
train_model(model,train_dl,NO_EPOCHS)


#%%
'''
plt.figure(figsize=(12,5))

plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(train_losses,label='Training loss')
plt.legend(frameon=False);

'''