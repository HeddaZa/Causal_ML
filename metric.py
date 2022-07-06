'''class for erupt metric'''

# TO DO:
    # repair ERUPT last method

from zlib import DEF_BUF_SIZE
import pandas as pd
import numpy as np

class extendedERUPT:
    def __init__(
        self,
        best_treatment: pd.Series,
        Treatment_test: pd.Series,
        target_test: pd.Series
    ):
        ''' 
        creates DataFrame with best_treatment,
        adds column with treatment of test set for each specific row
        
        Parameters
        ----------
        best_treatment: pd.Series
            best treatment option from models
        Treatment_test: pandas series
            treatment series of test set
        '''
        self.df_metric = pd.DataFrame({'best treatment':best_treatment})
        self.df_metric['test_set_treatment'] = Treatment_test
        self.df_metric['test_set_treatment'] = self.df_metric['test_set_treatment'].astype(int)
        self.df_metric["target test"] = target_test

    def _add_random_treatment_col(
        self,
        random_state:int
        ):
        '''
        adds column with shuffled values of column 'best treatment', which is the column
        displaying the best treatment options according to the model
        
        Parameters
        ----------
        random_state : int
        '''
        self.df_metric['random'] = self.df_metric['best treatment'].sample(n=self.df_metric.shape[0],replace = False, random_state = random_state).values
    

    def _get_common_treatment_index(
        self,
        col_to_compare:str
        )->pd.Series:
        '''
        filters for rows where treatment equals treatment of test set and 
        returns index of these rows
        
        Parameters
        ----------
        col_to_compare: string
            name of column to compare 'test_set_treatment' to

        Returns
        -------
        pandas series
            index of rows where treatments of both columns are equal
        '''
        return self.df_metric.loc[self.df_metric[col_to_compare] == self.df_metric['test_set_treatment']].index

    @staticmethod
    def p_value(
        distribution:np.ndarray, 
        value:float
        ) -> float:
        '''
        computes the p value
        
        Parameters
        ----------
        distribution: numpy array
        value: float
        
        Returns
        -------
        float
        p-value
        '''
        distribution = np.array(distribution)
        mean = np.mean(distribution)
        if value >= mean:
            outliers = len(distribution[distribution>value])*100/len(distribution)
        elif value < mean:
            outliers = len(distribution[distribution>value])*100/len(distribution)
        else:
            raise ValueError('input not valid')
        return outliers

    def _evaluate_random_distr(self):
        erupt_bench_values = []
        erupt_bench_sum_sucess = []
        erupt_bench_sum_match = []
        
        for xx in range(500):
            self._add_random_treatment_col(xx)
            index_erupt_bench = self._get_common_treatment_index('random')
            erupt_value_bench = self.df_metric["target test"].loc[index_erupt_bench].sum()/self.df_metric["target test"].loc[index_erupt_bench].shape[0]
            
            erupt_bench_values.append(erupt_value_bench)
            erupt_bench_sum_sucess.append(self.df_metric["target test"].loc[index_erupt_bench].sum())
            erupt_bench_sum_match.append(self.df_metric["target test"].loc[index_erupt_bench].shape[0])
        
        return np.mean(erupt_bench_values), np.mean(erupt_bench_sum_sucess), np.mean(erupt_bench_sum_match), erupt_bench_values


    def get_ERUPT_with_benchmark(
        self
        ):
        '''
        computes the ERUPT metric for the model and a randomised benchmark
        
        Parameters
        ----------
        prob_0: numpy array
            probability prediction of treatment 0
        prob_1: numpy array
            probability prediction of treatment 1
        prob_2: numpy array
            probability prediction of treatment 2
        X_test_index: pandas series
            index of test set
        Treatment_test: pandas series
            treatment column of test set
        y_test: pandas series
            target of test set
        
        Returns
        -------
        tuple 
            the erupt metric of the model, mean of outcome of test set, distribution of ERUPT of randomised model treatment
        '''
        index_erupt_test = self._get_common_treatment_index('best treatment')
        erupt_value_test = self.df_metric["target test"].loc[index_erupt_test].sum()/self.df_metric["target test"].loc[index_erupt_test].shape[0]

        erupt_bench_values_mean, erupt_bench_sum_sucess_mean, erupt_bench_sum_match_mean, erupt_bench_values = self._evaluate_random_distr()
        
        
        erupt_value_bench_2 = self.df_metric["target test"].mean()
        
        print(f'ERUPT model: {erupt_value_test:.4f}, sum of y: {self.df_metric["target test"].loc[index_erupt_test].sum()}, number of rows: {self.df_metric["target test"].loc[index_erupt_test].shape[0]}')
        
        print(f'ERUPT benchmark: {erupt_bench_values_mean:.4f}, sum of y: {round(erupt_bench_sum_sucess_mean)}, number of rows: {round(erupt_bench_sum_match_mean)}')
        print(f'Difference of ERUPT(model) and ERUPT(benchmark): {erupt_value_test - erupt_bench_values_mean:.4f}')
        
        return erupt_value_test, erupt_value_bench_2, erupt_bench_values
    
