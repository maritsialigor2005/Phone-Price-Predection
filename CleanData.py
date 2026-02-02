import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def CleanData():
    
    test= pd.read_csv("Data/test.csv")
    train= pd.read_csv("Data/train.csv")
    unCleanTest = pd.read_csv("Data/test.csv")
    unCleanTrain = pd.read_csv("Data/train.csv")

    cols_to_drop = [
        'Vo5G',# constant (1 value)
        'Dual_Sim',# most are the same
        '4G',# most are the same
        'RAM Tier',# ram size is more imp
        'Performance_Tier',# cpu is more imp
        'os_name',# brand name says the os anyways
        'os_version',# doesnt matter??
        'Processor_Series',# processor series doesnt dtermine speed 
        'Resolution_Height',# correlated with width
    ]

    # drop the columns that are not useful
    train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    test.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # handle outliers        
    for col in train.select_dtypes(include=['float64']).columns:
        if col != 'price':
            Q1 = train[col].quantile(0.25)
            Q3 = train[col].quantile(0.75)
            IQR = Q3 - Q1
        
        # di 3ashan el booleans byb2a feehom 0 IQR
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            train[col] = train[col].clip(lower=lower_bound, upper=upper_bound)
            
            

    # replacing 'Unknown' with NaN values so we cann handle easier
    train.replace('Unknown', np.nan, inplace=True)
    test.replace('Unknown', np.nan, inplace=True)

    pd.set_option('future.no_silent_downcasting', True)

    # replacing values in categorical columns with numerical codes
    # List of columns to encode
    categorical_cols = test.select_dtypes(include=['object']).columns


    for col in categorical_cols:
        le = LabelEncoder()
        combined_data = pd.concat([train[col], test[col]], axis=0).astype(str)
        codes, uniques = pd.factorize(combined_data)
        
        train[col] = codes[:len(train)]
        test[col] = codes[len(train):]
        
        # factorize turned NaNs into -1. 
        # turn them back to NaN so median works.
        train[col] = train[col].replace(-1, np.nan)
        test[col] = test[col].replace(-1, np.nan)


    # fill null values with median
    test = test.fillna(test.median(numeric_only=True))
    train.fillna(train.median(numeric_only=True), inplace=True)

    # remove the dupes
    test.drop_duplicates(inplace=True)
    train.drop_duplicates(inplace=True)

    # normalization
    for col in test.select_dtypes(include=['number']).columns:
        if col != 'price':
            test[col] = (test[col] - test[col].min()) / (test[col].max() - test[col].min())
    for col in train.select_dtypes(include=['number']).columns:
        if col != 'price':
            train[col] = (train[col] - train[col].min()) / (train[col].max() - train[col].min())
            
    test.to_csv("CleanData/CleanTest.csv", sep=',', index=False)
    train.to_csv("CleanData/CleanTrain.csv", sep=',' , index=False)

    print("Data Cleaned")

    data = {
        "numOfUncleanCols" : len(unCleanTest.columns),
        "numOfCleanCols" : len(test.columns),
        "numOfUncleanRowsInTest" : len(unCleanTest),
        "numOfUncleanRowsInTrain" : len(unCleanTrain),
        "numOfCleanRowsInTest" : len(test),
        "numOfCleanRowsInTrain" : len(train),
        "droppedCols" : cols_to_drop
    }

    return data
    