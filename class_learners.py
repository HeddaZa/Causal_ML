from random import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

        
        
    def _run_predictions(self, data):
        clf = self.classifier
        clf.fit(self.X_train, self.y_train)
        for j in range(self.number_of_treatments):
            self.proba_per_segment.append(clf.predict_proba(
                data[j][:,1]
            )
            )



class SLearner(Learner):  # unabhaengig von daten
    def __init__(
        self,
        file_name,
        classifier,
        test_split=0.5,
        random_state=42
    ):
        super().__init__(file_name)
        self.test_split = test_split
        self.random_state = random_state
        self.X_train = self.y_train = None
        self.X_test = self.y_test = None
        self.classifier = classifier
        
        self.X_per_segment = {}
        self.proba_per_segment = {}
        self.features()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data.drop(columns=["visit"]),
            self.data["visit"],
            test_size=self.test_split,
            random_state=self.random_state,
        )

    @staticmethod
    def s_learner_segment(data, option):
        cols = ["segment_0", "segment_1", "segment_2"]
        for i, col in enumerate(cols):
            data[col] = 0
            if i == option:
                data[col] = 1
        return data

    def _prepare_segments_test(self):
        keys_segment = ["segment_0_1", "segment_1_1", "segment_2_1"]
        for j in range(3):
            self.X_per_segment[keys_segment[j]] = self.s_learner_segment(
                self.X_test, j
            ).copy()

    def _run_predictions(self):
        keys_segment = ["segment_0_1", "segment_1_1", "segment_2_1"]
        clf = self.classifier
        clf.fit(self.X_train, self.y_train)
        for j in range(3):
            self.proba_per_segment[keys_segment[j]] = clf.predict_proba(
                self.X_per_segment[keys_segment[j]][:,1]
            )
   