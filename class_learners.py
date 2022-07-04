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
        test_split=0.5,
        random_state=42
    ):
        treatment = pd.DataFrame(treatment).astype('category')
        dummy_treatment = pd.get_dummies(treatment)
        self.dummy_name = dummy_treatment.columns
        features_and_treatment = pd.concat([features, dummy_treatment,treatment],axis =1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features_and_treatment,
            target,
            test_size=test_split,
            random_state=random_state
            )
        self.proba_per_segment = None

    def s_learner_segment(self, data, option):
        for i, col in enumerate(self.dummy_name):
            data[col] = 0
            if i == option:
                data[col] = 1
        return data

    def _split_test_set(self):
        self.X_test_per_segment = {}
        self.X_test[self.dummy_name] = 0
        for i,name in enumerate(self.dummy_name):
            self.X_test_per_segment[name] = self.s_learner_segment(
                self.X_test, i
            ).copy()

        
    # write better doc string   
    # inheritances: S learner + T learner to corrST, learner class with metric to all others 
    # function for drop segment
    



class SLearner(Learner):  
    def __init__(
        self,
        features, 
        treatment,
        target,
        test_split=0.5,
        random_state=42
    ):
        super().__init__(features, treatment, target, test_split, random_state)      
        self.X_test_per_segment = None 
        self.X_train.drop(columns = [treatment.name], inplace = True)
        self.X_test.drop(columns = [treatment.name], inplace = True)

    def _run_predictions(self, classifier, **kwrds):
        self.proba_per_segment = {}
        clf = classifier(**kwrds)
        if classifier == xgb.XGBClassifier:
            eval_set = [(self.X_train,self.y_train),(self.X_test,self.y_test)]
            clf.fit(self.X_train,self.y_train, eval_set = eval_set, verbose  = False)
        else:
            clf.fit(self.X_train, self.y_train)
        for  name in self.dummy_name:
            self.proba_per_segment[name] = clf.predict_proba(
                self.X_test_per_segment[name]
            )[:,1]

    def get_proba(self,classifier, **kwrds):
        self._split_test_set()
        self._run_predictions(classifier, **kwrds)
        return self.proba_per_segment


class TLearner(Learner):
    def __init__(
        self,
        features, 
        treatment,
        target,
        test_split=0.5,
        random_state=42
    ):
        super().__init__(features, treatment, target, test_split, random_state) 
        self.X_train.drop(columns = self.dummy_name, inplace = True)
        self.X_test.drop(columns = self.dummy_name, inplace = True)
        self.X_per_segment = None

    def _prepare_data_T(self):
        self.X_per_segment = {}

        for j, name in enumerate(self.dummy_name):
            self.X_per_segment[name] = self.segment_split_T(self.X_train, self.y_train,j)

    @staticmethod
    def segment_split_T(X_:pd.DataFrame,y_:pd.Series,segment:int):
        '''
        splits data into 3 parts according to segment
        
        Parameters:
        -----------
        X_: pd dataframe
            Dataframe without target but with segment
        y_: pd series
            target (with index)
        segment: int or string
            entry to filter for

        returns:
            X_split: pd dataframe
                X_ filtered by segemtn
            y_split: pd series
            y_ filtered by segment
        '''
        X_split = X_[X_['segment'] == segment].copy()
        y_split = y_.loc[X_split.index].copy()
        return X_split, y_split

    def _run_all_predictions(self, classifier, treatment_name, **kwrds):
        self.proba_per_segment = {}
        X_test_wo_tr = self.X_test.drop(columns = [treatment_name])

        for key in self.dummy_name:
            self.X_per_segment[key][0].drop(columns = [treatment_name], inplace = True)
            clf = self._run_predictions(classifier, self.X_per_segment[key][0], self.X_per_segment[key][1],**kwrds)

            self.proba_per_segment[key] = clf.predict_proba(
                X_test_wo_tr
            )[:,1]

    def _run_predictions(self, classifier,X_train, y_train, **kwrds):
        clf = classifier(**kwrds)
        if classifier == xgb.XGBClassifier:
            clf.fit(X_train,y_train, verbose  = False)
        else:
            clf.fit(X_train, y_train)
        return clf
        
    def get_proba(self,classifier, treatment_name,  **kwrds):
        self._prepare_data_T()
        self._run_all_predictions(classifier, treatment_name, **kwrds)
        return self.proba_per_segment   

class CorrSTLearner(Learner):
    def __init__(
        self,
        features, 
        treatment,
        target,
        test_split=0.5,
        random_state=42
    ):
        super().__init__(features, treatment, target, test_split, random_state)      
        self.X_test_per_segment = None 
        self.X_per_segment = None
        self.y_per_segment = None
        self.X_train.drop(columns = [treatment.name], inplace = True)
        self.X_test.drop(columns = [treatment.name], inplace = True)
    

    def _filter_and_split(self):
        self.X_per_segment = {}
        self.y_per_segment = {}

        for name in self.dummy_name:
            self.X_per_segment[name] = self.X_train[self.X_train[name] == 1].copy()
            self.y_per_segment[name] = self.y_train.loc[self.X_per_segment[name].index].copy()

    def prepare_data(self):
        self._filter_and_split()
        self._split_test_set()

    def _run_base_model(self, **kwrds):
        model_base = xgb.XGBClassifier(**kwrds)
        model_base.fit(self.X_train,self.y_train, verbose = False)
        model_base.save_model('model_base.model')

    def _run_predictions(self,X_train, y_train, **kwrds):
        self._run_base_model(**kwrds)
        clf = xgb.XGBClassifier(**kwrds)    
        clf.fit(X_train,y_train, verbose  = False, xgb_model='model_base.model')    
        return clf

    def _run_all_predictions(self, **kwrds):
        self.proba_per_segment = {}
        keys_segment = self.dummy_name

        for key in keys_segment:
            clf = self._run_predictions(self.X_per_segment[key], self.y_per_segment[key],**kwrds)

            self.proba_per_segment[key] = clf.predict_proba(
                self.X_test_per_segment[key]
            )[:,1]

    def get_proba(self, **kwrds):
        self._run_all_predictions( **kwrds)
        return self.proba_per_segment 

