# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:55:17 2018

Vamos a juntar aquí las instrucciones para hacer las imágenes para Bilbao

@author: Izzy
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from scipy.stats import chi2
from openpyxl import load_workbook
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
def compute_meanAndError(df,variable='Exhalation (Bq m- 2h-1)',
                         variable_error=None,weighted=False,sigma=1):
    """
    This function takes in a dataframe and a variable and computes the mean and
    error for the variable passed.
    For the computation, the function will calculate the external and internal 
    errors (i.e. standard deviations and internal errors, if provided).
    If asked, the function will compute the weighted mean and error and 
    multiply the error for a sigma parameter.
    """
    # Make sure variable is in the columns of df
    assert variable in df.columns
    # if variable_error is provided, check it as well
    if variable_error:
        assert variable_error in df.columns
    # Compute the mean
    if not weighted:
        avg = df[variable].mean()
        # Compute the standard deviation
        error = df[variable].std()
        # If provided, compute the internal error
        if variable_error:
            error_int = 1/np.sqrt(np.sum(np.square(1/df[variable_error])))
            # Add to the error list
            error = [error,error_int]
        # Pick the maximum of the errors and compute the uncertainty associated
        # to the mean (i.e. error/sqrt(n))
        error = sigma*np.max(error)/np.sqrt(len(df[variable]))
        return (avg,error)
    else:
        # Make sure a variable error is provided
        assert variable_error is not None
        # Compute the weighted mean
        avg = np.sum(df[variable]/df[variable_error])/np.sum(1/df[variable_error])
        # Compute the weighted standard deviation
        numerator = np.sum(np.square((df[variable]-avg))/np.square(df[variable_error]))
        denominator = (len(df[variable])-1)*np.sum(np.square(1/df[variable_error]))
        error = np.sqrt(numerator/denominator)
        # Compute the internal error
        error_int = 1/np.sqrt(np.sum(np.square(1/df[variable_error])))
        # Combine them
        error = [error,error_int]
        # Pick the maximum of the errors and compute the uncertainty associated
        # to the mean (i.e. error/sqrt(n))
        error = sigma*np.max(error)/np.sqrt(len(df[variable]))
        return (avg,error)
# =============================================================================

# =============================================================================
def filter_experiments(df,filt,values_to_avoid=None):
    """
    This function takes a dataframe, 'df', and a list of tuples, 'filt', and
    returns a Dataframe with the entries that coincide with the filter 
    conditions.
    
    The filter list must contains tuples of the form (ColumnName,ValueDesired),
    where 'ColumnName' is a valid name of a column of the DataFrame 'df' and 
    'ValueDesired' is the value that we want to find.
    
    The values_to_avoid is a list of tuples with the form
    [(Variable_1,[Value_1,Value_2,...]),
     (Variable_2,[Value_1,Value_2,...]),...]
    """
    # Check if filt is a 2-tuple, turning it into a list
    if type(filt) == tuple and len(filt) == 2:
        filt = [filt]
    # Initialize the index
    idx = []
    # Iterate over the contents of the filter
    for (column,value) in filt:
        # If it is the first iteration, the idx is empty
        if len(idx) == 0:
            # Fill the index
            idx = df[column] == value
        # In any other case, combine indexes gathered until now
        else:
            idx = idx & (df[column] == value)
    # Filter experiments
    if values_to_avoid:
        # Start index
        idx_avoid = []
        # extract the column and the forbidden values
        for column, avoid_list in values_to_avoid:
            # Iterate over the forbidden values
            for value in avoid_list:
                # If it is the first iteration, the idx is empty
                if len(idx_avoid) == 0:
                    # Fill the index
                    idx_avoid = df[column] == value
                # In any other case, combine indexes gathered until now
                else:
                    idx_avoid = idx_avoid | (df[column] == value)
        # Combine both indices
        idx = idx & ~idx_avoid
    # Return desired value
    return df[idx]
# =============================================================================

# =============================================================================
def retrieve_experiments_list(df,columns_to_retrieve,
                              values_to_avoid=None):
    """
    This function takes in a DataFrame, df, and a list of strings with the 
    names of the DataFrames columns that the user wants to check.
    The function will return a list of tuples of the form:
        [(column_1,unique_value_1),
         (column_1,unique_value_2),
         ...,
         (column_2,unique_value_1),
         (column_2,unique_value_2),
         ...]
    This list can be used with the function 'filter_experiments' to choose a
    subgroup of the DataFrame to do statistics.
    If 'values_to_avoid' is provided values contained within the list will be 
    ignores and not added to the final list of experiments.
    """
    # Check if columns_to_retrieve is a single value
    if type(columns_to_retrieve) != list:
        # Turn it into an iterable list
        columns_to_retrieve = [columns_to_retrieve]
    else:
        # Make sure that every column is in the Dataframe
        for column in columns_to_retrieve:
            assert column in df.columns
    # Initialize the list
    unique_experiments = []
    # Iterate over the columns
    for column in columns_to_retrieve:
        # Take the unique options in a list form
        column_options = df[column].unique().tolist()
        # Create a list of tuples with the experiments to check  
        if not values_to_avoid:
            experiments = [(column,entry) for entry in column_options]
        else:
            experiments = [(column,entry) for entry in column_options 
                            if entry not in values_to_avoid]
        # Add the experiments to the final list
        unique_experiments += experiments
    # Return the result
    return unique_experiments
# =============================================================================

# =============================================================================
def compute_zScore(variable_1,variable_2):
    """
    This function take two tuples with the average and the error of two 
    variables and computes the z-score between them.
    """
    # Make sure we are dealing with tuples
    assert type(variable_1) == type(variable_2) == tuple
    # Compute the z-score
    denominator = np.sqrt(variable_1[1]**2+variable_2[1]**2)
    return np.abs((variable_1[0]-variable_2[0])/denominator)
# =============================================================================

# =============================================================================
def compute_wide_zScore(df,column_to_test,
                        shared_features=[('Method','CC'),('Soil','Cnt')],
                        variable='E (Bqm-2h-1)',variable_error='sE (Bqm-2h-1)',
                        experiments_to_avoid=['V15ne','V35ne','VUPC']):
    """
    This function takes in a dataframe and a list of shared features. The 
    function will reduce the dataframe to all entries with the common 
    characteristics provided. Then it will perform a zScore for each 
    combination of the variables in the 'column_to_test' variable.
    
    The function will return the results in a table of the form:
              Cnt   V10   V15
        Cnt   0.5   3.6   2.5
        V10   0.23  3.2   1.8
        V15   0.56  3.8   2.9
    """
    # Find indices for the shared features and reduce the dataframe
    df_shared = filter_experiments(df,shared_features)
    # Retrieve a list of the experiments to test
    experimentList = retrieve_experiments_list(df_shared,column_to_test,
                                               experiments_to_avoid)
    # Initialize a list for the zscore
    list_zScore = []
    # And two list to keep track of the combinations
    list_exp1 = []
    list_exp2 = []
    # Iterate over the experiment's list
    for exp1 in experimentList:
        # Compute the mean and error
        var1 = compute_meanAndError(filter_experiments(df_shared,exp1),
                                    variable,variable_error)
        # Iterate again over the experiment's list
        for exp2 in experimentList:
            # Compute the mean and error
            var2 = compute_meanAndError(filter_experiments(df_shared,exp2),
                                                  variable,variable_error)
            # Save the experiment info
            list_exp1.append(exp1[1])
            list_exp2.append(exp2[1])
            # Compute the zscore and save it
            list_zScore.append(compute_zScore(var1,var2))
    # Create a new Dataframe to store results
    df_zScore = pd.DataFrame()
    # Guardo los nombres de experimentos
    col1 = column_to_test + ' (1)'
    col2 = column_to_test + ' (2)'
    df_zScore[col1] = list_exp1
    df_zScore[col2] = list_exp2
    # save results
    df_zScore['zScore'] = list_zScore
    # Return the pivot table
    return df_zScore.pivot_table(values='zScore',index=col1,columns=col2)
# =============================================================================
