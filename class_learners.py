'''Classes for S-, T-, and correlated ST-Learner, as well as a "Learner" parent class'''

#TO DO:
    # write doc strings and comments  
    # function for drop segment
    # double check similarities of T and ST: simplify with Learner method if possible
    # add try raise where appropriate
    # rename methods if appropriate

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

class Learner:
    '''
    General Learner class that is the parent class to all specific Learners in this file. It handles the initial data preparation 
    (train, test split and treatment preparation). Methods that will be used in more than one of the specific Learners will be
    hosted by the Learner class.

    Parameters:
    ---------------
    features: pd.DataFrame
        DataFrame with features (without treatment)
    treatment: pd.Series
        Series with the treatment values
    target: pd.Series
        Series with the target values
    test_split: float, default 0.5
        fraction for size of test set
    random_state: int, default 42
        root of randomisation of test/train split
    '''
    def __init__(
        self,
        features, 
        treatment,
        target,
        test_split=0.5,
        random_state=42
    ):
        #creating dummies for treatment
        treatment = pd.DataFrame(treatment).astype('category')
        dummy_treatment = pd.get_dummies(treatment)
        self.dummy_name = dummy_treatment.columns

        # concating features, dummy treatments, and original treatments to be split into test and train sets
        features_and_treatment = pd.concat([features, dummy_treatment,treatment],axis =1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features_and_treatment,
            target,
            test_size=test_split,
            random_state=random_state
            )

        self.test_treatment = self.X_test[treatment.columns[0]]
        self.proba_per_segment = None
        self.X_test_per_segment = None
        self.X_per_segment = None

    def s_learner_segment(self, data, option):
        '''
        setting all values of the DataFrame "data" to 0 but for column number "option", which is set to 1

        Parameters:
        -------------
        data: pd.DataFrame
            dummy DataFrame of treatment
        option: int
            the number of the column which will be set to 1

        returns:
        ------------
        data: pd.DataFrame
            dummy DataFrame with all values set to 0 but for column number "option", which is set to 1
        '''
        for i, col in enumerate(self.dummy_name):
            data[col] = 0
            if i == option:
                data[col] = 1
        return data

    def _split_test_set(self):
        '''
        setting all values of the DataFrame "data" to 0 but for column number "option", which is set to 1

        Parameters:
        -------------
        data: pd.DataFrame
            dummy DataFrame of treatment
        option: int
            the number of the column which will be set to 1
        '''
        self.X_test_per_segment = {}
        self.X_test[self.dummy_name] = 0
        for i,name in enumerate(self.dummy_name):
            self.X_test_per_segment[name] = self.s_learner_segment(
                self.X_test, i
            ).copy()

    def map_dummy_name(self):
        '''
        creates dictionary that maps digits to dummy names
        '''
        temp_dict = {}
        for j, name in enumerate(self.dummy_name):
            temp_dict[name]=j

        return temp_dict

    def get_best_treatment(
        self
        ) -> pd.Series:
        '''
        returns a dataframe with the treatment option that has the highest probability
        
        Returns:
        --------
        Series with one column with best treatment option and index of test set
        '''
        df_prob = pd.DataFrame(self.proba_per_segment)
        prob_max_col = df_prob.idxmax(axis = 1)
        prob_max_col.index = self.X_test.index
        dummy_dict = self.map_dummy_name()
        return prob_max_col.map(dummy_dict)

class SLearner(Learner): 
    '''
    S-learner class

    Parameters:
    ---------------
    features: pd.DataFrame
        DataFrame with features (without treatment)
    treatment: pd.Series
        Series with the treatment values
    target: pd.Series
        Series with the target values
    test_split: float, default 0.5
        fraction for size of test set
    random_state: int, default 42
        root of randomisation of test/train split
    ''' 
    def __init__(
        self,
        features, 
        treatment,
        target,
        test_split=0.5,
        random_state=42
    ):
        super().__init__(features, treatment, target, test_split, random_state)
        #dropping the treatment columns (as the DataFrame contains the treatment as dummies as well)
        self.X_train.drop(columns = [treatment.name], inplace = True)
        self.X_test.drop(columns = [treatment.name], inplace = True)

    def _run_predictions(self, classifier, **kwrds):
        '''
        instantiate classifier object and computes probabilities for each test set
        
        Parameters:
        -----------
        classifier: ml classifier
            classifier to be used
        **kwrds: 
            arguments for classifier
        '''
        self.proba_per_segment = {}
        clf = classifier(**kwrds)
        if classifier == xgb.XGBClassifier:
            clf.fit(self.X_train,self.y_train, verbose  = False)
        else:
            clf.fit(self.X_train, self.y_train)
        for  name in self.dummy_name:
            self.proba_per_segment[name] = clf.predict_proba(
                self.X_test_per_segment[name]
            )[:,1]

    def get_proba(self,classifier, **kwrds):
        '''
        performs all necessary methods to obtain probabilities
        
        Parameters:
        -----------
        classifier: ml classifier
            classifier to be used
        **kwrds: 
            arguments for classifier

        Returns:
        -----------
        pd.Series with treatment with highest probability per row
        '''
        self._split_test_set()
        self._run_predictions(classifier, **kwrds)
        return self.get_best_treatment()      


class TLearner(Learner):
    '''
    T-learner class

    Parameters:
    ---------------
    features: pd.DataFrame
        DataFrame with features (without treatment)
    treatment: pd.Series
        Series with the treatment values
    target: pd.Series
        Series with the target values
    test_split: float, default 0.5
        fraction for size of test set
    random_state: int, default 42
        root of randomisation of test/train split
    ''' 
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
        

    def _prepare_data_T(self):
        '''
        filters the train data set for the different treatment options and creates a dictionairy for the new DataFrames
        '''
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
        '''
        instantiate a classifier object per train data set and computes proabilities on the test set
        
        Parameters:
        -----------
        classifier: ml classifier
            classifier to be used
        treatment_name: str
            name of the treatment column
        **kwrds: 
            arguments for classifier
        '''
        self.proba_per_segment = {}
        X_test_wo_tr = self.X_test.drop(columns = [treatment_name])

        for key in self.dummy_name:
            self.X_per_segment[key][0].drop(columns = [treatment_name], inplace = True)
            clf = self._run_predictions(classifier, self.X_per_segment[key][0], self.X_per_segment[key][1],**kwrds)

            self.proba_per_segment[key] = clf.predict_proba(
                X_test_wo_tr
            )[:,1]

    def _run_predictions(self, classifier,X_train, y_train, **kwrds):
        '''
        fits classifier object with train data
        
        Parameters:
        -----------
        classifier: ml classifier
            classifier to be used
        X_train: pd.DataFrame
            train data set
        y_train: pd.Series
            target series associated with the train data set
        **kwrds: 
            arguments for classifier
        '''
        clf = classifier(**kwrds)
        if classifier == xgb.XGBClassifier:
            clf.fit(X_train,y_train, verbose  = False)
        else:
            clf.fit(X_train, y_train)
        return clf
        
    def get_proba(self,classifier, treatment_name,  **kwrds):
        '''
        performs all necessary methods to obtain probabilities
        
        Parameters:
        -----------
        classifier: ml classifier
            classifier to be used
        treatment_name: str
            name of the treatment column
        **kwrds: 
            arguments for classifier

        Returns:
        --------
        Series with one column with best treatment option and index of test set
        '''
        self._prepare_data_T()
        self._run_all_predictions(classifier, treatment_name, **kwrds)
        return self.get_best_treatment()    

class CorrSTLearner(Learner):
    '''
    correlated ST-learner class

    Parameters:
    ---------------
    features: pd.DataFrame
        DataFrame with features (without treatment)
    treatment: pd.Series
        Series with the treatment values
    target: pd.Series
        Series with the target values
    test_split: float, default 0.5
        fraction for size of test set
    random_state: int, default 42
        root of randomisation of test/train split
    ''' 
    def __init__(
        self,
        features, 
        treatment,
        target,
        test_split=0.5,
        random_state=42
    ):
        super().__init__(features, treatment, target, test_split, random_state)      
        self.y_per_segment = None
        self.X_train.drop(columns = [treatment.name], inplace = True)
        self.X_test.drop(columns = [treatment.name], inplace = True)
    

    def _filter_and_split(self):
        '''
        filters train data (both features and target) for treatment option and adds both the created DataFrames and Series to a dictionaries.
        '''
        self.X_per_segment = {}
        self.y_per_segment = {}

        for name in self.dummy_name:
            self.X_per_segment[name] = self.X_train[self.X_train[name] == 1].copy()
            self.y_per_segment[name] = self.y_train.loc[self.X_per_segment[name].index].copy()

    def prepare_data(self):
        '''
        prepares test and train set for correlated ST-Learner
        '''
        self._filter_and_split()
        self._split_test_set()

    def _run_base_model(self, **kwrds):
        '''
        instantiates classifiert object and runs base model
        
        Parameters:
        -----------
        classifier: ml classifier
            classifier to be used
        **kwrds: 
            arguments for classifier
        '''
        model_base = xgb.XGBClassifier(**kwrds)
        model_base.fit(self.X_train,self.y_train, verbose = False)
        model_base.save_model('model_base.model')

    def _run_predictions(self,X_train, y_train, **kwrds):
        '''
        instantiate a classifier object. Fits model with boost from a base model. 
        Computes proabilities on the test set
        
        Parameters:
        -----------
        X_train: pd.DataFrame
            train feature data
        y_train: pd.Series
            train target data
        **kwrds: 
            arguments for classifier
        '''
        self._run_base_model(**kwrds)
        clf = xgb.XGBClassifier(**kwrds)    
        clf.fit(X_train,y_train, verbose  = False, xgb_model='model_base.model')    
        return clf

    def get_proba(self, **kwrds):
        '''
        computes the best treatment using the correlated ST-Learner

        Parameters:
        --------------
        **kwrds:
            parameters for xbgoost classifier
       
        Returns:
        --------
        Series with best treatment option and index of test set
        '''
        self.proba_per_segment = {}
        keys_segment = self.dummy_name

        for key in keys_segment:
            clf = self._run_predictions(self.X_per_segment[key], self.y_per_segment[key],**kwrds)
            self.proba_per_segment[key] = clf.predict_proba(
                self.X_test_per_segment[key]
            )[:,1]
        return self.get_best_treatment() 


