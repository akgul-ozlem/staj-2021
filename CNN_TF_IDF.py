#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 09:23:29 2021

@author: oa
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

import torch
import torchtext
import scipy
import torch.nn.functional as F

# my files
#import news_preprocess

#%% read hphyperparameters 
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
NO_EPOCHS     = 15              # Tuned parameters for ttc3600 cleaned
LEARNING_RATE = 0.001
MOMENTUM      = 0.9
KERNEL_SIZE   = 10
KERNEL_SIZE1  = 10
KERNEL_SIZE2  = 3
POOLING       = 3
DROPOUT       = 0.17
STRIDE        = 1
BATCH_SIZE    = 128            # or 64

# random state
RS = 42
torch.manual_seed(RS)
torch.cuda.manual_seed(RS)
np.random.seed(RS)
# %% import data
data = pd.read_excel(file_name, page)            #, encoding = 'utf-8')

cat_hdr  = 'cat'
data_hdr = 'description'
test_ratio = 0.2

# summarize dataset
print("input data shape:   ", data.shape)
print(data.groupby(cat_hdr).size())
print(data[cat_hdr].nunique())

le = LabelEncoder().fit(data[cat_hdr])
data['encodedcat'] = le.transform(data[cat_hdr])

train_news, test_news, train_topics, test_topics = train_test_split(data[data_hdr],data['encodedcat'],test_size=test_ratio,stratify=data[cat_hdr],random_state=RS)

#%%  prepare data

vectorizer = TfidfVectorizer()                              #Tf-Idf vectorization 
x_train = vectorizer.fit_transform(train_news)              #fit-transform learns the vocabulary
tf_len= len(vectorizer.vocabulary_)
#print(tf_len)
x_test = vectorizer.transform(test_news)                    #tranform uses the vocabulary and frequencies learned by fit_transform 
#print(vectorizer.get_feature_names())                       #print the learned vocabulary

scaler1 = StandardScaler(with_mean=0).fit(x_train)      #Scaling train dataset
x_train_scaled = scaler1.transform(x_train)

scaler2 = StandardScaler(with_mean=0).fit(x_test)       #Scaling test dataset
x_test_scaled = scaler2.transform(x_test)

x_train = torch.tensor(scipy.sparse.csr_matrix.todense(x_train_scaled)).float()  #returns the dense representation of the sparse matrix x_train_scaled and converts it to a tensor
x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test_scaled)).float()

#x_train = torch.tensor(scipy.sparse.csr_matrix.todense(x_train)).float()  #returns the dense representation of the sparse matrix x_train_scaled and converts it to a tensor
#x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()

y_train = torch.tensor(train_topics.values)                 #converts the panda object to a tensor 
y_test = torch.tensor(test_topics.values)

#print(x_train.shape)
#x_train=x_train[:,:20000]
#x_test =x_test[:,:20000]
#print(x_train.shape)
#print(x_test.shape)

training_tuples = []
testing_tuples = []

#print(x_train.shape[0])
i=0
while i < x_train.shape[0]:
    tuple1=(x_train[i,:],y_train[i])
    training_tuples.append(tuple1)
    i=i+1

k=0

while k < x_test.shape[0]:
    tuple2=(x_test[k,:],y_test[k])
    testing_tuples.append(tuple2)
    k=k+1

NUM_LABEL = data[cat_hdr].nunique()
TEST_BATCH = y_test.shape[0]

train_dl= DataLoader(training_tuples,batch_size=64,shuffle=True)
test_dl = DataLoader(testing_tuples, batch_size =TEST_BATCH, shuffle = True)

#https://cezannec.github.io/CNN_Text_Classification/
#https://github.com/cezannec/CNN_Text_Classification/blob/master/CNN_Text_Classification.ipynb

#%%
#https://www.machinecurve.com/index.php/2021/03/29/batch-normalization-with-pytorch/

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.LazyConv1d(1, KERNEL_SIZE,STRIDE)
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

'''
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #self.dropout1 = nn.Dropout2d(0.5)
        self.conv1=nn.LazyConv1d(5, KERNEL_SIZE1,STRIDE)
        self.pool1=nn.MaxPool1d(POOLING)
        self.conv2 = nn.LazyConv1d(1,KERNEL_SIZE2,STRIDE)
        self.pool2 = nn.MaxPool1d(2)
        self.hidden = nn.Linear(13731,32)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.hidden2 = nn.Linear(32,6)
        
    def forward(self,x):
        #x=self.dropout1(x)
        x=self.conv1(x)    
        x=F.relu(x)
        x=self.pool1(x)
        x=F.relu(self.conv2(x)) 
        x=self.pool2(x)   
        x=x.view(x.size(0),-1) 
        x=self.dropout2(self.hidden(x))
        x= self.hidden2(x)
        output=F.softmax(x,dim=1)
        return output
''' 
 

#https://www.youtube.com/watch?v=8YsZXTpFRO0
#https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412/2
        
model=Net()

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
        optimizer.zero_grad()
        for train_data, train_label_data in train_dl: 

            train_data=train_data.unsqueeze(1)    #[batch_size, in_channels, len]

            output = model.forward(train_data)
            loss = criterion(output,train_label_data)
            
            loss.backward()
            train_loss = loss.item()
            train_losses.append(train_loss)
            optimizer.step()
            
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
            test_data=test_data.unsqueeze(1)    
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
