
import sys
import pandas as pd
import os
import numpy as np
sys.path.append('.')


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score

from framework import framework, dataloader, LR

sys.path.append('../')

struct_log = 'C:/Harshad.Ambekar/personal/github/hackathon-predict-system/dataset/log_structured.csv' 
label_file = 'C:/Harshad.Ambekar/personal/github/hackathon-predict-system/dataset/anomaly_label.csv'


if __name__ == '__main__':


    (x_train, y_train), (x_test, y_test) = dataloader.load_dataset(struct_log,
                                                            label_file=label_file,
                                                            window='session', 
                                                            train_ratio=0.5,
                                                            split_type='uniform')   

    feature_extractor = framework.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='sigmoid')    
    x_test = feature_extractor.transform(x_test)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    print('====== Evaluation summary ======')
    print('Train validation:')    
    y_train_pred = model.predict(x_train)
    
    precision = precision_score(y_train_pred, y_train)
    recall = recall_score(y_train_pred, y_train)
    print(precision)
    print(recall)
    #print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))

    print('Test validation:')    
    #model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    #print(y_test_pred)
    precision = precision_score(y_test_pred, y_test )
    recall = recall_score(y_test, y_test_pred)
    print(precision)
    print(recall)
    #precision, recall, f1, = precision_recall_fscore_support(x_test, y_test_pred, average='binary')
    #print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
    
