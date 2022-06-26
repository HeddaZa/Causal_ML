import pandas as pd
import numpy as np

    
def treatment_category(
    series:pd.Series
    )-> pd.Series:
    '''maps 'No E-Mail', 'Womens E-Mail','Mens E-Mail' onto 0, 1, 2
        
    Parameters:
    -----------
    series: pandas series
    
    Returns:
    --------
    a series with mapped entries
    '''
    treatments = ['No E-Mail', 'Womens E-Mail','Mens E-Mail']
    dictionary = dict(zip(treatments, list(range(3))))
    return series.map(dictionary)


def standardise(
    series:pd.Series
    )->pd.Series:
    '''
    standardises column *1/2
            
    Parameters:
    -----------
    series: pandas series
    
    Returns:
    --------
    a series with standardises entries
    '''
    mean = np.mean(series)
    std = np.std(series)
    return (series-mean)/(std*2)

def get_features(
    df:pd.DataFrame, 
    include_treatment:bool = False
    )->pd.DataFrame:
    '''
    takes features and one-hot-encodes them. if include_treatment is True, 
    treatment column will be added to features
            
    Parameters:
    -----------
    df: pandas DataFrame
        
    include_treatment: cool
        default = False
        if True: column 'segment' will be hot-encoded
    
    Returns:
    --------
    pandas DataFrame
    '''
    features1 = ['mens', 'womens', 'newbie']
    features2 = [  'zip_code', 'channel']
    if include_treatment:
        features2 = features2+['segment']
    df[features1+features2] = df[features1+features2].astype('category')
    dummy_features = pd.get_dummies(df[features2])
    df.drop(columns = features2, inplace = True)
    df = pd.concat([df,dummy_features], axis = 1)

   
    return df

def feature_engineering(
    df:pd.DataFrame, 
    include_treatment:bool = False
    )->pd.DataFrame:
    '''
    performs all necessary feature engineering tasks
            
    Parameters:
    -----------
    df: pandas DataFrame
        
    include_treatment: cool
        default = False
        if True: column 'segment' will be hot-encoded
    
    Returns:
    --------
    pandas DataFrame
    '''
    df.drop(columns= ['history_segment', 'conversion','spend'], inplace = True)

    df['recency'] = standardise(df['recency'])
    df['history'] = standardise(df['history'])

    df['segment'] = treatment_category(df['segment'])
    df = get_features(df, include_treatment=include_treatment)
    
    return df

    




def read_in_data(file_name,include_treatment = False):
    '''
    reads in data
    
    Parameters:
    -----------
    file_name: string
        filename of file to be read in
         
    returns:
        X: pd dataframe
            dataframe with features
        y: pd series
            pd series with target
    '''
    
    Hillstrom_path = file_name
    Hillstrom = pd.read_csv(Hillstrom_path)
    
    Hillstrom = feature_engineering(Hillstrom, include_treatment = include_treatment)

    y = Hillstrom['visit']
    X = Hillstrom.drop(columns = 'visit')
    
    return X,y
    
