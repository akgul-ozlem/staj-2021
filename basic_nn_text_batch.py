#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:02:15 2021

@author: kaan
"""

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

import torch
import torchtext
import scipy

# my files
#import news_preprocess

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
NO_EPOCHS = 75
LEARNING_RATE = 0.01
MOMENTUM = 0.9





# %% import data
data = pd.read_excel(file_name, page)            #, encoding = 'utf-8')

# summarize dataset
print("input data shape:   ", data.shape)
print(data.groupby('class').size())


le = LabelEncoder().fit(data["class"])
data['encodedcat'] = le.transform(data['class'])

train_news, test_news, train_topics, test_topics = train_test_split(data['description'],data['encodedcat'],test_size=.2,stratify=data['class'],random_state=42)

#%%  prepare data


vectorizer = TfidfVectorizer()                              #Tf-Idf vectorization 
x_train = vectorizer.fit_transform(train_news)              #fit-transform learns the vocabulary

x_test = vectorizer.transform(test_news)                    #tranform uses the vocabulary and frequencies learned by fit_transform 
#print(vectorizer.get_feature_names())                       #print the learned vocabulary


scaler1 = StandardScaler(with_mean=False).fit(x_train)      #Scaling train dataset
x_train_scaled = scaler1.transform(x_train)

scaler2 = StandardScaler(with_mean=False).fit(x_test)       #Scaling test dataset
x_test_scaled = scaler2.transform(x_test)

x_train = torch.tensor(scipy.sparse.csr_matrix.todense(x_train_scaled)).float()  #returns the dense representation of the sparse matrix x_train_scaled and converts it to a tensor
x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test_scaled)).float()


y_train = torch.tensor(train_topics.values)                 #converts the panda object to a tensor 
y_test = torch.tensor(test_topics.values)



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


train_dl= DataLoader(training_tuples,batch_size=2880,shuffle=False)
test_dl = DataLoader(testing_tuples, batch_size =720, shuffle = False)

print(x_train.shape[1])


#%% training function

train_losses = []
test_losses = []
test_accuracies = []
val_outputs= []


#model = nn. Sequential(nn.Linear(x_train.shape[1],data['class'].nunique()),nn.ReLU(),nn.Dropout(0.5),nn.Softmax(dim=1))
model = nn. Sequential(nn.Linear(x_train.shape[1],64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,data['class'].nunique()),nn.ReLU(),nn.Dropout(0.2),nn.Softmax(dim=1))
criterion = nn.CrossEntropyLoss()

def train_model(model,train_dl,epochs):
    model.train()
    
    optimizer= optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM)
    for epoch in range(epochs):
        optimizer.zero_grad()

        for train_data, train_label_data in train_dl: 
            
            output = model(train_data)
            loss = criterion(output,train_label_data)
            loss.backward()
            train_loss = loss.item()
            train_losses.append(train_loss)
            optimizer.step()
        print(f"Epoch: {epoch+1}/{epochs}..", f"Training loss: {train_loss:.3f}")

def evaluate_model(model,test_dl):
    model.eval()
    actual = []
    for test_data,test_label_data in test_dl:
            
        val_out = model(test_data)
        val_outputs.append(val_out)
        print(val_out.shape)
        val_loss= criterion(val_out,test_label_data)
        test_losses.append(val_loss)
        targets=test_label_data.numpy()
                    
        _, predictions = torch.max(val_out,dim=1) 
        accuracy = accuracy_score(predictions, targets)
        test_accuracies.append(accuracy)
        actual.append(predictions) #Creates a list with all the outputs
        f1_score_result=f1_score(test_label_data,predictions,average='weighted')     
    print(f"Validation loss: {val_loss:.3f}" ,f"Validation accuracy:{accuracy:.3f}")
     
    print(f"F1 socre: {f1_score_result:.3f}")


print("Training....")
train_model(model,train_dl,NO_EPOCHS)


print("Validation....")
evaluate_model(model,test_dl)

#%%
plt.figure(figsize=(12,5))

plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(train_losses,label='Training loss')
plt.legend(frameon=False);

