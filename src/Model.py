# -*- coding: utf-8 -*-
"""
Stephanie Yow
Personal Practice
Description: Hotel Booking No-Show Prediction (Model.py)
"""

import sqlite3
import pandas as pd

import Preprocess
import Resample

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

# extract dataset
connect = sqlite3.connect('data/noshow.db')
df = pd.read_sql_query('SELECT * FROM noshow;', connect)
connect.close()

# preproceess columns
df = Preprocess.drop_empty_rows(df)
df = Preprocess.arrival_month(df)
df = Preprocess.booking_month(df)
df = Preprocess.checkout_month(df)
df = Preprocess.months_stayed(df)
df = Preprocess.checkout_day(df)
df = Preprocess.nights_stayed(df)
df = Preprocess.price(df)
df = Preprocess.price_per_night(df)
df = Preprocess.branch(df)
df = Preprocess.drop_columns(df)
df = Preprocess.one_hot_encoding(df)

# rebalance dataset
df_downsample = Resample.resampled(df)
# randomly shuffle dataset
df_final = df_downsample.sample(frac = 1, 
                                random_state = 42, 
                                ignore_index = True)

# set features
features = df_final.drop(columns = ['no_show'])
# feature Z-score normalisation
scaler = StandardScaler()
scaler.fit(features)
scaled_features = scaler.fit_transform(features)

# set target
y = df_final['no_show']

# split dataset into training and test data
x_train, x_test, y_train, y_test = train_test_split(scaled_features, y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)

# fit model
model = LogisticRegression(solver = 'liblinear')
model.fit(x_train, y_train)

# predict target values
predictions = model.predict(x_test)
predict_proba = model.predict_proba(x_test)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
loss = log_loss(y_test, predict_proba)

# evaluation report
report = pd.DataFrame({'Metrics': ['Accuracy Score', 'F1 Score', 'Log Loss'], 
                       ' ': [accuracy, f1, loss]})
report.set_index('Metrics')
report.to_csv('evaluation_report.txt', sep = '\t', index = False)