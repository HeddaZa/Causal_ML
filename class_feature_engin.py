import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#import model_metric as mm


class feature_engineering:  # gross camel case
    def __init__(self, file_name: str) -> None:
        self.data = pd.read_csv(file_name)
        self.data.drop(columns=["history_segment", "conversion", "spend"], inplace=True)

    def _treatment_category(self) -> None:
        """maps column 'visit's values 'No E-Mail', 'Womens E-Mail','Mens E-Mail' onto 0, 1, 2"""
        treatments = list(
            self.data["segment"].value_counts().index
        )  # Womens, Mens, No E-Mail
        dictionary = dict(zip(treatments, list(range(3))))
        self.data["segment"] = self.data["segment"].map(dictionary)

    @staticmethod
    def standardise(series: pd.Series) -> pd.Series:
        """
        standardises column *1/2

        Parameters:
        -----------
        series: pandas series

        Returns:
        --------
        a series with standardises entries
        """
        mean = np.mean(series)
        std = np.std(series)
        return (series - mean) / (std * 2)

    def _get_features(self):
        """
        takes features and one-hot-encodes them. if include_treatment is True,
        treatment column will be added to features
        """
        features1 = ["mens", "womens", "newbie"]
        features2 = ["zip_code", "channel", "segment"]
        self.data[features1] = self.data[features1].astype("category")
        self.data[features2] = self.data[features2].astype("category")
        dummy_features2 = pd.get_dummies(self.data[features2])
        self.data.drop(columns=features2, inplace=True)
        self.data = pd.concat([self.data, dummy_features2], axis=1)
        self.data["recency"] = self.standardise(self.data["recency"])
        self.data["history"] = self.standardise(self.data["history"])

    def features(self):
        """
        performs all necessary feature engineering procedures

        Parameters:
        -----------
        self

        returns:
            X: pd dataframe
                dataframe with features
            y: pd series
                pd series with target
        """
        self._treatment_category()
        self._get_features()
        return self.data


class S_learner(feature_engineering):  # unabhaengig von daten
    def __init__(
        self,
        file_name,
        classifier,
        test_split=0.5,
        random_state=42,
        **params
    ):
        super().__init__(file_name)
        self.test_split = test_split
        self.random_state = random_state
        self.X_train = self.y_train = None
        self.X_test = self.y_test = None
        self.classifier = classifier
        self.params = params
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
        clf = self.classifier#(self.params)
        clf.fit(self.X_train, self.y_train)
        for j in range(3):
            self.proba_per_segment[keys_segment[j]] = clf.predict_proba(
                self.X_per_segment[keys_segment[j]][:,1]
            )
   