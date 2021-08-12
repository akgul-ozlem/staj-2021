#%% imports
import fasttext
import numpy as np
import csv
import pickle
import os

from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, recall_score,
from sklearn.metrics import f1_score, accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#%% start timer
from timeit import default_timer as timer
import time 
start = timer()
print('---------------------------------------------------------------------')
print('started @      ', time.asctime())

def elapsed_time(start_time):
    now = timer()
    print('now:         ', time.asctime())
    elapsed_tm = now - start_time
    print('elapsed time: %0.d sec' % (elapsed_tm))    
    print('---------------------------------------------------------------------')
    return elapsed_tm

#%% read pickle for traning FastText
fasttext_vec_file  = '/okoksal/ws-meta/fasttext_lib/cc.tr.300.vec'           # D:\okoksal\ws-meta\fasttext_lib
# fasttext_vec_file  = '/okoksal/ws-meta/fasttext_lib/wiki-news-300d-1M.vec' # tükrçe bug verisetinde çalışmıyor
# fasttext_vec_file  = '/okoksal/ws-meta/fasttext_lib/crawl-300d-2M.vec'     # tükrçe bug verisetinde çalışmıyor

print('---------------------------------------------------------------------')
print('reading FastText training vectors ...')

#check if pkl file exists
pkl_filename = fasttext_vec_file + '.pkl'
if os.path.exists(pkl_filename):
    print('reading pickle file...')
    with open(pkl_filename, "rb") as f:
        vectors = pickle.load(f)
# if not exist 
else:
    # read vector file first
    with open(fasttext_vec_file, 'r', encoding='utf8') as f:
        print('reading vec file...')
        vectors = {}
        for row in f:
            row = row.split()
            vectors[row[0]] = [float(i) for i in row[1:]]
    # then generate a pkl file
    with open(pkl_filename, "wb") as f:
        print('writing pickle')
        pickle.dump(vectors, f)

elapsed_time(start)

#%% read hyperparameters
file_name     = "data/data_new_ok.csv"  # "data/bug_data_ok.csv"
# file_name     = 'data/1-ttc3600_noSW_noLem.csv'
# file_name     = 'data/2-ttc4900_noSW_noLem.csv'

out_file      = "results/bug_results"   #
rs            = 42                      # random state, max_value @rs=1536 
rf_estimators = 200                     # # of estimaters used in RF
test_ratio    = 0.2
CV            = 10                      # 10 fold - cross validation
vector_length = 300
metric        = 'accuracy'              # accuracy, f1-weighted

print('- files -------------------------------------------------------------')
print('file_name           ', file_name)
print('out_file            ', out_file)
print('fasttext_vec_file   ', fasttext_vec_file)
print('- hyper parameters  -------------------------------------------------')
print('random state        ', rs)
print('test_ratio          ', test_ratio)
print('rf_estimators       ', rf_estimators)
print('cross validation    ', CV)
print('vector_length       ', vector_length)
print('metric              ', metric)
print('---------------------------------------------------------------------')

#%% select possible classifier
clsfr = []
clsfr.append(GaussianNB())
clsfr.append(SVC(gamma='scale', kernel='linear'))
clsfr.append(SVC(gamma='scale', kernel='poly'))
clsfr.append(SVC(gamma='scale', kernel='rbf'))
clsfr.append(SVC(gamma='scale', kernel='sigmoid'))
clsfr.append(KNeighborsClassifier(n_neighbors=17))
clsfr.append(LogisticRegression(solver='lbfgs', multi_class='ovr',fit_intercept=True)) #, max_iter = 200)
clsfr.append(DecisionTreeClassifier())
clsfr.append(RandomForestClassifier(n_estimators = rf_estimators, criterion = 'gini'))
print('classifiers:\n', clsfr)

#%% main program

# read dataset
for datapath, respath in zip([file_name], [out_file]):
    with open(datapath, "r") as f:
        r = csv.reader(f)
        data, label = [], []
        first=True
        for row in r:
            if first:
                first=False
            else:
                label.append(row[0])
                data.append(row[1].replace("_", " "))

    vector_data = np.zeros((len(data), vector_length), dtype=np.float64)

for clf in clsfr:
    print('classifier           -', clf) 
    for normalization in [True, False]:
        for i, row in enumerate(data):
            row = row.split()
            text = np.zeros((len(row), vector_length), dtype=np.float64)
            for j, w in enumerate(row):
                try:
                    text[j] = vectors[w]
                except KeyError:
                    continue
            mean = np.average(text, axis=0)
            if normalization:
                vector_data[i] = mean / np.linalg.norm(mean, ord=2)
            else:
                vector_data[i] = mean

        results = []
        for fold in range(CV):
            # print('fold: ', fold, '-', 'normalization: ', normalization, ' - rnd state - ', rndStt)
            train_data, test_data, train_label, test_label = train_test_split(vector_data, label, test_size=test_ratio, random_state=rs)
            clf.fit(train_data, train_label)
            pred = clf.predict(test_data)

            # p = precision_score(test_label, pred, average="weighted")
            # r = recall_score(test_label, pred, average="weighted")
            f = f1_score(test_label, pred, average="weighted")
            a = accuracy_score(test_label, pred)  # , acc için average='weighted' diye bir opsiyon yok!

            # print("Precision: {}, \nRecall:    {}, \nF1 Score:        {}, \nAccuracy:  {}".format(p ,r, f, a))
            # print('---------------------------------------------------------------------')
            if(metric == 'accuracy'):
                results.append(a)                
            else:
                results.append(f)

        if normalization:
            with open(respath + "_Acc_normalized.txt", "w") as f:
                f.write("10-Fold accuracy\n")
                f.write(str(np.average(results)))
                print('normalization:', normalization, ' - np.average(results): ', np.average(results))
                # print('---------------------------------------------------------------------')
        else:
            with open(respath + "_Acc.txt", "w") as f:
                f.write("10-Fold accuracy\n")
                f.write(str(np.average(results)))
                print('normalization:', normalization, '- np.average(results): ', np.average(results))
                print('---------------------------------------------------------------------')
                
#%% print completion time
print('---------------------------------------------------------------------')
elapsed_time(start)
