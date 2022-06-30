from random import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
#import model_metric as mm

class Learner:
    def __init__(
        self,
        features, 
        treatment,
        target,
        classifier,
        test_split=0.5,
        random_state=42
        ) -> None:

        self.classifier = classifier
        self.proba_per_segment = []
        features_and_treatment = pd.concat([features,pd.DataFrame(treatment)], axis = 1)
        self.number_of_treatments = treatment.nunique()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features_and_treatment,
            target,
            test_size=test_split,
            random_state=random_state
            )

        
    #define data better
    # write better doc string    
    def _run_predictions(self, data):
        '''
        data is list with datasets (= numbers of treatments)
        '''
        clf = self.classifier
        clf.fit(self.X_train, self.y_train)
        for j in range(self.number_of_treatments):
            self.proba_per_segment.append(clf.predict_proba(
                data[j][:,1]
            )
            )



class SLearner:  
    def __init__(
        self,
        features, 
        treatment,
        target,
        test_split=0.5,
        random_state=42
    ):
        self.treatment_dummy = pd.get_dummies(treatment)
        features_and_treatment = pd.concat([features, self.treatment_dummy],axis =1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features_and_treatment,
            target,
            test_size=test_split,
            random_state=random_state
            )
        self.X_per_segment = None
        self.proba_per_segment = None
    
    
    def s_learner_segment(self, data, option):
        cols = self.treatment_dummy.columns
        for i, col in enumerate(cols):
            data[col] = 0
            if i == option:
                data[col] = 1
        return data


    def _prepare_segments_test(self):
        self.X_per_segment = {}
        keys_segment = ["segment_0_1", "segment_1_1", "segment_2_1"]
        for j in range(3):
            self.X_per_segment[keys_segment[j]] = self.s_learner_segment(
                self.X_test, j
            ).copy()

    def _run_predictions(self, classifier, **kwrds):
        self.proba_per_segment = {}
        keys_segment = ["segment_0_1", "segment_1_1", "segment_2_1"]
        clf = classifier(**kwrds)
        if classifier == xgb.XGBClassifier:
            eval_set = [(self.X_train,self.y_train),(self.X_test,self.y_test)]
            clf.fit(self.X_train,self.y_train,eval_metric="logloss", eval_set = eval_set)
        else:
            clf.fit(self.X_train, self.y_train)
        for j in range(3):
            self.proba_per_segment[keys_segment[j]] = clf.predict_proba(
                self.X_per_segment[keys_segment[j]]
            )[:,1];

    def get_proba(self,classifier, **kwrds):
        self._prepare_segments_test()
        self._run_predictions(classifier, **kwrds)
        return self.proba_per_segment


        