import sys
from typing import List

import inflection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    
    '''
    The function optimize_floats takes a pandas DataFrame as input and optimizes the memory usage of float columns in the DataFrame. 
    It first selects all float columns using select_dtypes method with include=['float64'] parameter, and converts them to the smallest 
    possible float type using pd.to_numeric method with downcast='float' parameter. Finally, the function returns the modified DataFrame.
    '''
    
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    
    '''
    The function optimize_ints takes a pandas DataFrame as input and optimizes the memory usage of integer columns in the DataFrame. 
    It first selects all integer columns using select_dtypes method with include=['int64'] parameter, and converts them to the smallest 
    possible integer type using pd.to_numeric method with downcast='integer' parameter. Finally, the function returns the modified DataFrame. 
    '''
        
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    
    '''
    The function optimize_objects takes a pandas DataFrame as input, along with a list of column names (datetime_features) that are supposed to be 
    of datetime type. The function optimizes the memory usage of object columns in the DataFrame.

    It first loops over all object columns in the DataFrame using select_dtypes method with include=['object']. For each column, if it is not in 
    datetime_features and not of type list, the function checks if the ratio of the number of unique values to the total number of values in the 
    column is less than 0.5. If it is, the column is converted to the 'category' data type, which can save memory compared to object data type.

    If a column is in datetime_features, it is converted to datetime type using pd.to_datetime method. Finally, the function returns the modified DataFrame.
    '''
    
    
    for col in df.select_dtypes(include=['object']):
        if col not in datetime_features:
            if not (type(df[col][0])==list):
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if float(num_unique_values) / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_datetime(df[col])
    return df


def optimize(df: pd.DataFrame, datetime_features: List[str] = []):
    
    '''
    The function optimize takes a pandas DataFrame as input, along with an optional list of column names (datetime_features) that are supposed to be of
    datetime type. The function optimizes the memory usage of integer, float, and object columns in the DataFrame using the optimize_ints, optimize_floats,
    and optimize_objects functions.

    The function first calculates the memory usage of the input DataFrame using sys.getsizeof method. It then calls the three optimization functions in the
    following order: optimize_objects, optimize_ints, and optimize_floats.

    After the optimization is complete, the function prints out the percentage of memory reduction achieved by the optimization. 
    Finally, the function does not return anything, neverthelss it change the types of passed DataFrame.
    '''
    
    mem_usage_bef = sys.getsizeof(df)
    optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))
    print(f'Optimize_memory_func reduce memory usage by {(1-(sys.getsizeof(df)/mem_usage_bef))*100 :.2f} %.')
    
    
def snake_case_col(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Utils for personal preferences.
    Change column names into snake case type.
    '''
    df.columns = df.columns.map(lambda x: inflection.underscore(x))
    print(f'Columns name have changed sucessfully: {list(df.columns.values[:3])}...')