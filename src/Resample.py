# -*- coding: utf-8 -*-
"""
Stephanie Yow
Personal Practice
Description: Hotel Booking No-Show Prediction (Rebalance.py)
"""

import pandas as pd
from sklearn.utils import resample

def resampled(X):
    '''
    Rebalance dataset by downsampling and then upweighting majority class
    '''
    # split dataset into majority and minority classes
    majority_class = X[X['no_show'] == 0.0]
    minority_class = X[X['no_show'] == 1.0]

    # downsample majority class
    majority_downsample = resample(majority_class, 
                                   replace = False, 
                                   n_samples = len(minority_class), 
                                   random_state = 42)
            
    # form new dataset of downsampled majority and original minority classes
    X_downsample = pd.concat([minority_class, majority_downsample])
            
    # upweight majority class
    original_weight = len(majority_class) / len(X)
    downsampling_factor = len(majority_class) / len(majority_downsample)
    upweight = original_weight * downsampling_factor
    
    # add a new column for weight
    X_downsample['weight'] = [upweight if ele == 0.0 else 0.0 \
                               for ele in X_downsample['no_show']]
    return X_downsample
