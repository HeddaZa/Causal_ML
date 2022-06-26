import pandas as pd 
import numpy as np 
import Quini as qi
import matplotlib.pyplot as plt

def get_best_treatment(
    prob_0:np.ndarray,
    prob_1:np.ndarray,
    prob_2:np.ndarray,
    X_test_index:np.ndarray
    ) -> pd.DataFrame:
    '''
    chooses the highest probability of three treatment options and returns a dataframe with
    the treatment option with the highest probability
    
    Parameters:
    -----------
    prob_0: numpy array
        probability prediction of treatment 0
    prob_1: numpy array
        probability prediction of treatment 1
    prob_2: numpy array
        probability prediction of treatment 1
    X_test_index: numpy array
        index of test set
    
    Returns:
    --------
    dataframe with one column with best treatment option and index of test set
    '''
    df_prob = pd.DataFrame(data = {'0':prob_0,'1':prob_1,'2':prob_2})
    prob_max_col = df_prob.idxmax(axis = 1)
    df_best_treatment = pd.DataFrame(data = {'prob_max_treat':prob_max_col})
    df_best_treatment = df_best_treatment.set_index(X_test_index)
    return df_best_treatment

def add_test_treatment(
    df:pd.DataFrame,
    X_test_index:np.ndarray,
    Treatment_test:pd.Series
    ) -> pd.DataFrame:
    ''' adds column with treatment of test set for each specific row
    
    Parameters
    ----------
    df : pandas dataframe
        dataframe to which the test treatment should be added
    X_test_index : numpy array
        index array of test set
    Treatment_test: pandas series
        treatment series of test set

    Returns
    -------
    pandas dataframe
        original dataframe df with additional column of test treatments for each specific row
    
    '''
    df['test_set_treatment'] = Treatment_test.loc[X_test_index]
    df['prob_max_treat'] = df['prob_max_treat'].astype(int)
    return df

def add_random_treatment_col(
    df:pd.DataFrame, 
    random_state:int
    )->pd.DataFrame:
    '''
    adds column with shuffled values of column 'prob_max_treat', which is the column
    displaying the best treatment options according to the model
    
    Parameters
    ----------
    df : pandas dataframe
        a dataframe with the column 'prob_max_treat'

    Returns
    -------
    dataframe
        original dataframe with added column of shuffled treatment options from model
    '''
    df['random'] = df['prob_max_treat'].sample(n=df.shape[0],replace = False, random_state = random_state).values
    return df
    

def get_common_treatment_index(
    df:pd.DataFrame,
    col_to_compare:str
    )->pd.Series:
    '''
    filters for rows where treatment equals treatment of test set and 
    returns index of these rows
    
    Parameters
    ----------
    df : pandas dataframe
        a dataframe with columns 'test_set_treatment' and column to compare to
    col_to_compare: string
        name of column to compare 'test_set_treatment' to

    Returns
    -------
    pandas series
        index of rows where treatments of both columns are equal
    '''
    return df.loc[df[col_to_compare] == df['test_set_treatment']].index

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


def get_ERUPT_with_benchmark(
    prob_0:np.ndarray,
    prob_1:np.ndarray,
    prob_2:np.ndarray,
    X_test_index:np.ndarray,
    Treatment_test:pd.Series,
    y_test:pd.Series
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
    df_best_treatment = get_best_treatment(prob_0,prob_1,prob_2,X_test_index)
    df_best_treatment = add_test_treatment(df_best_treatment, X_test_index, Treatment_test)
    
    
    index_erupt_test = get_common_treatment_index(df_best_treatment, 'prob_max_treat')
    erupt_value_test = y_test.loc[index_erupt_test].sum()/y_test.loc[index_erupt_test].shape[0]
 
    erupt_bench_values = []
    erupt_bench_sum_sucess = []
    erupt_bench_sum_match = []
    
    for xx in range(500):
        df_best_treatment = add_random_treatment_col(df_best_treatment,xx)
        index_erupt_bench = get_common_treatment_index(df_best_treatment, 'random')
        erupt_value_bench = y_test.loc[index_erupt_bench].sum()/y_test.loc[index_erupt_bench].shape[0]
        
        erupt_bench_values.append(erupt_value_bench)
        erupt_bench_sum_sucess.append(y_test.loc[index_erupt_bench].sum())
        erupt_bench_sum_match.append(y_test.loc[index_erupt_bench].shape[0])
        
    erupt_bench_values_mean = np.mean(erupt_bench_values)
    erupt_bench_sum_sucess_mean = np.mean(erupt_bench_sum_sucess)
    erupt_bench_sum_match_mean = np.mean(erupt_bench_sum_match)
    
    
    erupt_value_bench_2 = y_test.mean()
    
    print(f'ERUPT model: {erupt_value_test:.4f}, sum of y: {y_test.loc[index_erupt_test].sum()}, number of rows: {y_test.loc[index_erupt_test].shape[0]}')
    
    print(f'ERUPT benchmark: {erupt_value_bench:.4f}, sum of y: {y_test.loc[index_erupt_bench].sum()}, number of rows: {y_test.loc[index_erupt_bench].shape[0]}')
    print(f'Difference of ERUPT(model) and ERUPT(benchmark): {erupt_value_test - erupt_value_bench:.4f}')
    
    print(f'ERUPT benchmark: {erupt_bench_values_mean:.4f}, sum of y: {round(erupt_bench_sum_sucess_mean)}, number of rows: {round(erupt_bench_sum_match_mean)}')
    print(f'Difference of ERUPT(model) and ERUPT(benchmark_2): {erupt_value_test - erupt_bench_values_mean:.4f}')
    
    return erupt_value_test, erupt_value_bench_2, erupt_bench_values
    



    

    
def prepare_for_qini(df_o,n,cols = ['segment','visit']):
    df = df_o[cols].copy()
    df[cols[0]] = df[cols[0]].astype('category')
    df_hec = pd.get_dummies(df[[cols[0]]])
    df = pd.concat([df,df_hec],axis = 1)
    df = df.astype(int)
    return df
    
def plot_qini_random(xs,ys):
    x = [0,xs[-1]]
    y = [0,ys[-1]]
    return plt.plot(x,y,label = 'random')   