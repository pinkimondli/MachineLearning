#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 07:58:45 2018

@author: andrea
"""

import pandas as pd
import numpy as np

class knnClassifier():
    #this implements a k-nearest-neighbour classifier
    def __init__(self,nNearestNeighbours=1):
        self.kNN = nNearestNeighbours

    def fit(self,dataset):
        self.dataset = dataset
        
    def predict(self,test_feature):
        # input: data-frame of test_data, the column containing the labels must be calles 'labels'
        # output: data-frame of (labels, features) according to majority-vote of k-nearest-neighbours
        distance_set = self._measure_distance(test_feature,self.dataset)
        prediction = test_feature.copy()
        for iTest in range(distance_set.shape[1]-1):
            distance_sorted = distance_set.sort_values(by=iTest)
            maj_vote_i = distance_sorted['label'][:self.kNN].value_counts().index[0]
            prediction.at[iTest,'label'] = maj_vote_i
        return prediction
        
    def _measure_distance(self,test_feature,train_data):
        # input: data-frames of feature values in test and train data
        # output: data-frame with (data = (euclidian distance), columns = (label of compared train_data,test_feature), index = index of train_data)
        distance_set = pd.DataFrame(train_data['label'])
        for iTest in range(test_feature.shape[0]):
            total_dist = pd.DataFrame()
            features=list(test_feature.columns)
            for feature in features:
                distance_to_f = train_data[feature]-test_feature[feature][iTest]
                total_dist[feature] = distance_to_f**2
            distance_set[iTest] = np.sqrt(total_dist.sum(axis=1))
        return distance_set
 
class decisionTreeClassifier():
    def __init__(self):
        pass
    
    def fit(self,dataset):
        features=list(dataset.columns)
        features.remove('label')
        
        decision_rule=[]
        siblings=[]
        best_feature = ''
        while len(features)>0:
            dataset=dataset[features]
            if len(siblings:)>0:
                dataset=dataset[dataset[best_feature].isin([siblings])][features]
            f_split = self._split_along_features(dataset,features)
            decision_rule.append(f_split['bestf_analysis'])
            if f_split['best_performance'] == 1:
                features = []
            if len(f_split['bestf_analysis'].index)>0:
                siblings = f_split['bestf_analysis'].index[1:]
            else:
                features=f_split['features']
        
        return decision_rule
        
    def predict(self,test_feature):
        pass
    
    def _calc_feature_accuracy(self,subset,feature):
        # input: subset to be analysed and current list of available features
        # return: accuracy obtained by splitting the set along this feature
        # do_notes: create dataset: index = uniqe examples of a feature; accuracy = counts of maj_vote of this example
        f_analysis=pd.DataFrame(index=subset[feature].unique())
        f_set = subset[feature]
        for example in list(f_analysis.index):
            maj_vote = subset[f_set.isin([example])]['label'].value_counts().index[0]
            f_analysis.at[example,'decision'] = maj_vote
            f_analysis.at[example,'accuracy'] = f_set.isin([example]).value_counts()[maj_vote]
        return sum(f_analysis['accuracy']/subset.shape[0]), f_analysis.sort_values(by='accuracy')
    
    def _split_along_features(self,subset,features):
        # input subset to be analysed and current list of available features        
        # return: feature along which splitting is most successful, its accuracy and new list of available features
        # do_notes:
        best_performance = 0
        best_feature = 'null'
        for feature in features:
            # 0. split along feature; 1. find which is maj_vote; 2. count how often these values occur
            f_accuracy,f_analysis = self._calc_feature_accuracy(subset,feature)
            if (f_accuracy > best_performance):
                best_performance = f_accuracy
                best_feature = feature
                bestf_analysis = f_analysis
        return dict({'best_feature' : best_feature, 'best_performance' : best_performance, 'bestf_analysis': bestf_analysis, 'features' : features.remove(best_feature)})
        