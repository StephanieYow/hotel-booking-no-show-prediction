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
Prediction of no-shows is a binary classification problem. There are only two possible classes, 0 (showed up) or 1 (no-show). From exploratory data analysis, hotel branch, booking month, arrival month, duration of stay, country of origin and average price per night were found to be reasonable predictors of no-shows.

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
The data is imbalanced in favour of class 0 (63% of records). If the data is not rebalanced, the resulting model may be better at predicting the majority class than minority class.

> Solution: Downsample the majority class to yield 44,224 samples to match the number of minority class records, and then upweight the majority class by creating a new column of weights. Newly balanced dataset should have 88,448 records in total.

The data is randomly shuffled to minimise bias.

### Standardisation
The data contains varying scales and undergoes Z-score normalisation.

### Modeling

After declaring feature and target variables, the data undergoes a train-test split with 0.2 test size. 
> x_train.shape: (70758, 13)
> 
> y_train.shape: (70758, 1)
> 
> x_test.shape: (17690, 13)
> 
> y_test.shape: (17690, 1)

Prediction of no-shows can be treated as a probability problem that requires output of binary values. Logistic regression is an efficient mechanism for calculating probabilities and is selected as the model.

After the model is fitted with training data, predictions are made on test data. Results are evaluated on a set of metrics: accuracy score, F1 score and log loss, and output to `evaluation_report.txt`. 

|Metric |Result|
|--|--|
|Accuracy Score|1.0|
|F1 Score|1.0|
|Log Loss|0.00014|

## Methodology Support
Here are more explanations of selected features. Please refer to `eda.ipynb` for the full exploratory data analysis.

### branch
66.4% of all bookings was for the Changi branch. When bookings were isolated to no-shows, the percentage jumped to 74.9%.

### country
40.7% of all bookings was from Chinese customers. When bookings were isolated to no-shows, the percentage jumped to 62.2%.

### arrival_month_no
Most no-shows were observed to arrive between April and August based on arrival_month. arrival_month_no is a numerical representation of arrival_month in order to be used in the model.
> Example:
> 
> January: 1.0
> 
> February: 2.0
> 
> March: 3.0
> 
> ...
> 
> November: 11.0
> 
> December: 12.0

### booking_month_no
Most no-shows were observed to be from bookings made in June and September based on booking_month. booking_month_no is a numerical representation of booking_month in order to used in the model.

### nights_stayed
Due to the absence of a date column in the raw data, nights_stayed is inferred. For example, if months_stayed is 1.0, and arrival_month is January, then nights_stayed = checkout_day - arrival_day + 31 days. Generally, the longer the stay, the less no-shows were observed. Longer stays may signal stronger intent to show up.

### price_per_night
price_per_night is closer to how hotels charge in the real world and is a fairer indication of intent to show up. For example, a customer with an SGD 1,000 booking for 5 nights is different from a customer with an SGD 1,000 booking for 2 nights. Customers whose bookings were SGD 750-1000 in total or SGD 375-500 and SGD 875-1000 per night had higher chance of not showing up.
