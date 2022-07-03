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
            clf.fit(self.X_train,self.y_train, eval_set = eval_set, verbose  = False)
        else:
            clf.fit(self.X_train, self.y_train)
        for j in range(3):
            self.proba_per_segment[keys_segment[j]] = clf.predict_proba(
                self.X_per_segment[keys_segment[j]]
            )[:,1]

    def get_proba(self,classifier, **kwrds):
        self._prepare_segments_test()
        self._run_predictions(classifier, **kwrds)
        return self.proba_per_segment


class TLearner:
    def __init__(
        self,
        features, 
        treatment,
        target,
        test_split=0.5,
        random_state=42
    ):
        features_and_treatment = pd.concat([features, treatment],axis =1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features_and_treatment,
            target,
            test_size=test_split,
            random_state=random_state
            )
        self.proba_per_segment = None  
        self.X_per_segment = {}
        number_of_unique_treatments = self.X_train[treatment.name].nunique()
        names_sections = ['train_0','train_1','train_2']

        for j in range(number_of_unique_treatments):
            self.X_per_segment[names_sections[j]] = self.segment_split_T(self.X_train, self.y_train,j)
            self.X_per_segment[names_sections[j]][0].drop(columns = [treatment.name], inplace = True)
       
        self.X_test.drop(columns = ['segment'], inplace = True)  


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

    def _prepare_segments(self,treatment_name):
        self.X_per_segment = {}
        number_of_unique_treatments = self.X_train[treatment_name].nunique()
        names_sections = ['train_0','train_1','train_2']

        for j in range(number_of_unique_treatments):
            self.X_per_segment[names_sections[j]] = self.segment_split_T(self.X_train, self.y_train,j)
            #self.X_per_segment[names_sections[j]][0].drop(columns = [treatment_name], inplace = True)
       
        self.X_test.drop(columns = ['segment'], inplace = True)

    def _run_all_predictions(self, classifier, **kwrds):
        self.proba_per_segment = {}
        keys_segment = ['train_0','train_1','train_2']

        for key in keys_segment:
            clf = self._run_predictions(classifier, self.X_per_segment[key][0], self.X_per_segment[key][1],**kwrds)

            self.proba_per_segment[key] = clf.predict_proba(
                self.X_test
            )[:,1]

    def _run_predictions(self, classifier,X_train, y_train, **kwrds):
        clf = classifier(**kwrds)
        if classifier == xgb.XGBClassifier:
            #eval_set = [(self.X_train,self.y_train),(self.X_test,self.y_test)]
            clf.fit(X_train,y_train, verbose  = False)
        else:
            clf.fit(X_train, y_train)
        return clf
        
    def get_proba(self,classifier, **kwrds):
        #self._prepare_segments(treatment_name=treatment_name)
        self._run_all_predictions(classifier, **kwrds)
        return self.proba_per_segment   

class CorrSTLearner:
    def __init__(
        self,
        features, 
        treatment,
        target,
        test_split=0.5,
        random_state=42
    ):
        self.number_of_unique_treatments = treatment.nunique()
        treatment = pd.DataFrame(treatment).astype('category')
        dummy_treatment = pd.get_dummies(treatment)
        features_and_treatment = pd.concat([features,dummy_treatment], axis = 1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features_and_treatment,
            target,
            test_size=test_split,
            random_state=random_state
            )
        self.X_per_segment = None
        self.X_test_per_segment = None
        self.dummy_name = dummy_treatment.columns

    def _filter_and_split(self):
        self.X_per_segment = {}
        #names_sections = ['train_0','train_1','train_2']

        for name in self.dummy_name:
            self.X_per_segment[name] = self.X_train[self.X_train[name] == 1].copy()

    def s_learner_segment(self, data, option): #inherite from SLearner
        cols = self.dummy_name
        for i, col in enumerate(cols):
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

    def prepare_data(self):
        self._filter_and_split()
        self._split_test_set()
        ### hier weiter machen

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
        keys_segment = ['train_0','train_1','train_2']

        for key in keys_segment:
            clf = self._run_predictions(self.X_per_segment[key][0], self.X_per_segment[key][1],**kwrds)

            self.proba_per_segment[key] = clf.predict_proba(
                self.X_test
            )[:,1]

    def get_proba(self, **kwrds):
        #self._prepare_segments(treatment_name=treatment_name)
        self._run_all_predictions( **kwrds)
        return self.proba_per_segment 

