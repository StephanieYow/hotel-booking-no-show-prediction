# -*- coding: utf-8 -*-
"""
Stephanie Yow
Personal Practice
Description: Hotel Booking No-Show Prediction (Preprocess.py)
"""

import pandas as pd

month_dictionary = {
    'January': 1.0,
    'February': 2.0,
    'March': 3.0,
    'April': 4.0,
    'May': 5.0,
    'June': 6.0,
    'July': 7.0,
    'August': 8.0,
    'September': 9.0,
    'October': 10.0,
    'November': 11.0,
    'December': 12.0
    }

def drop_empty_rows(X):
    '''
    Omit rows with null values in no_show column
    '''
    X = X[~X['no_show'].isnull()]
    return X

def arrival_month(X):
    '''
    Transform arrival_month column values to the same letter case and
    add a new column of float values representing the month of arrival
    '''
    X['arrival_month'] = X['arrival_month'].map(lambda x: x.capitalize())
    X['arrival_month_no'] = [month_dictionary[month] \
                                 for month in X['arrival_month'] \
                                     if month in month_dictionary.keys()]
    return X

def booking_month(X):
    '''
    Add a new column of float values representing the month of booking
    '''
    X['booking_month_no'] = [month_dictionary[month] \
                                 for month in X['booking_month'] \
                                     if month in month_dictionary.keys()]
    return X

def checkout_month(X):
    '''
    Add a new column of float values representing the month of checkout
    '''
    X['checkout_month_no'] = [month_dictionary[month] \
                                 for month in X['checkout_month'] \
                                     if month in month_dictionary.keys()]
    return X

def months_stayed(X):
    '''
    Add a new column of float values for the number of months stayed
    '''
    X['months_stayed'] = X['checkout_month_no'] - X['arrival_month_no']
    X['months_stayed'] = X['months_stayed'].map(lambda x: 1.0 \
                                               if x == -11.0 else x)
    return X

def checkout_day(X):
    '''
    Transform negative float values of checkout_day column to positive 
    '''
    X['checkout_day'] = X['checkout_day'].map(lambda x: -1 * x \
                                              if x < 0 else 1 * x)
    return X

def nights_stayed(X):
    '''
    Add a new column of float values for the number of nights stayed
    '''
    nights_stayed = []
    
    for i in range(len(X)):
        
        if X.iloc[i, 18] == 0.0:
            duration = X.iloc[i, 7] - X.iloc[i, 5]
            nights_stayed.append(duration)
    
        elif X.iloc[i, 18] == 1.0:
            if X.iloc[i, 4] in ('January', 'March', 'May', 'July', \
                                'August', 'October', 'December'):
                duration = X.iloc[i, 7] - X.iloc[i, 5] + 31
                nights_stayed.append(duration)
            
            elif X.iloc[i, 4] in ('April', 'June', 'September', 'November'):
                duration = X.iloc[i, 7] - X.iloc[i, 5] + 30
                nights_stayed.append(duration)
        
            else:
                duration = X.iloc[i, 7] - X.iloc[i, 5] + 28
                nights_stayed.append(duration)   
        
        elif X.iloc[i, 18] == 2.0:
            if X.iloc[i, 4] in ('January', 'March', 'May', 'July', \
                                'August', 'October', 'December'):
                duration = X.iloc[i, 7] - X.iloc[i, 5] + 31 + 30
                nights_stayed.append(duration)
            
            elif X.iloc[i, 4] in ('April', 'June', 'September', 'November'):
                duration = X.iloc[i, 7] - X.iloc[i, 5] + 30 + 31
                nights_stayed.append(duration)
                
            else:
                duration = X.iloc[i, 7] - X.iloc[i, 5] + 28 + 31
                nights_stayed.append(duration)
        
    X['nights_stayed'] = nights_stayed
    return X       

def price(X):
    '''
    Transform string values of price column into float values by removing
    currency prefixes and converting USD to SGD, and impute nulls with
    median price
    '''
    X['price'] = X['price'].fillna('0')
    
    X['price'] = X['price'].map(lambda x: float(x[5:]) \
                                if 'SGD$' in str(x) \
                                    else float(x[5:]) / 0.745 \
                                        if 'USD$' in str(x) else float(x))

    price_not_0 = X[~(X['price'] == 0.0)]
    median_price = price_not_0['price'].describe()['50%']
    
    X['price'] = X['price'].map(lambda x: x if x != 0.0 else median_price)
    return X

def price_per_night(X):
    '''
    Calculate average price per night stayed
    '''
    X['price_per_night'] = X['price'] / X['nights_stayed']
    return X

def branch(X):
    '''
    Transform categorical string values of branch column to representative
    float values
    '''
    X['branch'] = X['branch'].map(lambda x: 0.0 if x != 'Changi' else 1.0)
    return X

def drop_columns(X):
    '''
    Retain essential columns for modeling
    '''
    X = X[[
        'no_show',
        'branch',
        'booking_month_no',
        'arrival_month_no',
        'country',
        'price_per_night',
        'nights_stayed'
        ]]
    return X

def one_hot_encoding(X):
    '''
    Perform one hot encoding for categorical string values of country column
    '''
    X['country'] = X['country'].astype('category')
    return pd.get_dummies(X, columns = ['country'], dtype = float)