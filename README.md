## About
Fictional Sunshine Hotels in Singapore wants to formulate policies to reduce expenses incurred due to no-shows. They requested analysis of over 100,000 customer records to understand behaviour of past no-shows and predict future no-shows. Both problem statement and dataset belong to AI Singapore's AI Apprenticeship Programme (AIAP(R))'s technical assessment past-year series.

## Contents
Base folder contains
> Folder `src`
> 
> Jupyter notebook `eda.ipynb`
> 
> `requirements.txt`

Folder `src` contains
> Folder `data`
> 
> `Model.py`
> 
> `Preprocess.py`
> 
> `Resample.py`

Folder `data` contains
> `noshow.db`

`Preprocess.py` and `Resample.py` are imported and executed in `Model.py`, which contains the machine learning workflow to predict no-shows. Please refer to `eda.ipynb` for the full exploratory data analysis.

## Data extracted from `noshow.db`
There are 119,391 records and 15 columns.

| Column | Non-Null Count | Dtype | Example Values |
|--|--|--|--|
| booking_id | 119391 | int64 | 66947
| no_show | 119390 | float64 | 1.0
| branch | 119390 | object | Orchard, Changi
| booking_month | 119390 | object | September
| arrival_month | 119390 | object | October, OctOber, OctobeR
| arrival_day | 119390 | float64 | 15.0
| checkout_month | 119390 | object | November
| checkout_day | 119390 | float64 | 3.0, -3.0
| country | 119390 | object | China
| first_time | object | object | Yes
| room | 97778 | object | Single
| price | 94509 | object | USD$ 665.37, SGD$ 937.55
| platform | 119930 | object | Website
| num_adults | 119930 | object | 1, one
| num_children | 119930 | float64 | 0.0

## Methodology

### Exploratory Data Analysis
From exploratory data analysis, hotel branch, country of origin, arrival month and booking month were found to influence no-shows. 2 other influencing factors, nights stayed and price per night were not as straightforward, and had to be derived from existing columns relating to arrival/checkout and total booking price respectively.

#### branch
66.4% of all bookings was for the Changi branch. When bookings were isolated to no-shows, the percentage jumped to 74.9%.

#### country
40.7% of all bookings was from Chinese customers. When bookings were isolated to no-shows, the percentage jumped to 62.2%.

#### arrival_month
Most no-shows were observed to arrive between April and August.

#### booking_month
Most no-shows were observed to be from bookings made in June and September.

#### nights_stayed
Due to the absence of a date column, nights_stayed is inferred. For example, if months_stayed is 1.0, and arrival_month is January, then nights_stayed = checkout_day - arrival_day + 31 days. Generally, the longer the stay, the less no-shows were observed. Longer stays may signal stronger intent to show up.

#### price_per_night
price_per_night is how hotels charge in the real world and is a fairer indication of intent to show up. For example, a customer with an SGD 1,000 booking for 5 nights is different from a customer with an SGD 1,000 booking for 2 nights. Customers whose bookings were SGD 750-1000 in total or SGD 375-500 and SGD 875-1000 per night had higher chance of not showing up.

### Preprocessing
Here is an overview of treatments to existing and new columns. 

| Column | Treatment | Dtype |
|--|--|--|
| branch | Transform categorical string values to float | float64 |
| arrival_month | Standardise letter case | object |
| checkout_day | Transform any negative float values to positive | float64 |
| country | One hot encoding | float64 |
| price | Remove currency prefixes, cast string values to float and impute nulls with median price | float64 |
| arrival_month_no | New numbered column corresponding to arrival_month | float64 |
| booking_month_no | New numbered column corresponding to booking_month | float64 |
| checkout_month_no | New numbered column corresponding to checkout_month | float64 |
| months_stayed | New column derived from arrival_month_no subtracted from checkout_month_no that represents stay duration in months | float64 |
| nights_stayed | New column inferred from arrival_month, months_stayed, arrival_day and checkout_day that represents stay duration in nights | float64 | 
| price_per_night | New column derived from price divided by nights_stayed that represents average price per night | float64 |

Only branch, encoded country, arrival_month_no, booking_month_no, nights_stayed and price_per_night columns are retained at the end of preprocessing.

### Resampling
The data is moderately imbalanced in favour of class 0 (63% of records). If the data is not rebalanced, the resulting model may be better at predicting the majority class than minority class.

> Solution: Using resampling without replacement, downsample the majority class to yield 44,224 samples to match the number of minority class records, and then upweight the majority class by creating a new column of weights. Newly balanced dataset should have 88,448 records in total.

### Feature and Target Variables
After declaring feature and target variables, the data undergoes a train-test split with 0.2 test size. 
> X_train.shape: (70758, 13)
> 
> y_train.shape: (70758,)
> 
> X_test.shape: (17690, 13)
> 
> y_test.shape: (17690,)

### Feature Scaling
Features contain varying scales from 10s to 1000s, and undergo Z-score normalisation to standardise their range. Statistical properties (mean and standard deviation) of X_train are used to scale both X_train and X_test.

### Modelling
Prediction of no-shows can be modelled as a probability problem (how likely would a customer no-show?) where the label follows a binomial distribution with only two outcomes, failure ('show up') or success ('no-show'). With this in mind, logistic regression is an appropriate model choice. Logistic regression, as its name implies, first performs regression, and then compresses regression outputs into probabilities using the logistic function. The model learns the best regression feature weights that maximise P(y = 1) whenever the true label y is 1 ('no-show') and maximise 1 - P(y = 1) whenever the true label y is 0 ('show up').

After the model is fitted with X_train and y_train, predictions are made on X_test. Predictions are evaluated against y_test on a set of metrics: accuracy score, F1 score and log loss, and output to `evaluation_report.txt`. 

|Metric |Result|
|--|--|
|Accuracy Score|1.0|
|F1 Score|1.0|
|Log Loss|0.00018|
