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

### Preliminary Preprocessing and Exploratory Data Analysis
As seen from the preceding section, data had to first be preprocessed for null values, spelling, letter casing and formatting inconsistencies, and anomalous values, before conducting exploratory data analysis.

For example:
| Column | Treatment |
|--|--|
| arrival_month | Standardise letter case |
| checkout_day | Transform negative values to positive |
| price | Remove currency prefixes, cast string values to float and impute nulls with median price |

Comparisons were then made between customers that showed up and customers that no-showed in order to spot characteristics more prevalent for the latter. Eventually, (hotel) branch, country (of origin), arrival_month and booking_month were found to have distinguishing characteristics.

#### branch
66.4% of all bookings was for the Changi branch. When bookings were isolated to no-shows, the percentage jumped to 74.9%.

#### country
40.7% of all bookings was from Chinese customers. When bookings were isolated to no-shows, the percentage jumped to 62.2%.

#### arrival_month
Most no-shows were observed to arrive between April and August.

#### booking_month
Most no-shows were observed to be from bookings made in June and September.

It made sense within the context of hotel bookings to create and analyse two new columns, nights_stayed and price_per_night, from existing columns. In the real world, bookings are usually based on the number of nights stayed, and customers are charged per night.

#### nights_stayed
nights_stayed was inferred from arrival_month, months_stayed, arrival_day and checkout_day. For example, if months_stayed was 1.0, and arrival_month was January, then nights_stayed = checkout_day - arrival_day + 31 days (number of days in January). Generally, the longer the stay, the less no-shows were observed.

Computing nights_stayed involved some intermediate steps.

| Step | Column | Treatment |
|--|--|--|
| 1 | arrival_month_no | New numerical column corresponding to arrival_month |
| 2 | checkout_month_no | New numerical column corresponding to checkout_month |
| 3 | months_stayed | New numerical column derived from arrival_month_no subtracted from checkout_month_no that represents stay duration in months |
| 4 | nights_stayed | New numerical column inferred from arrival_month, months_stayed, arrival_day and checkout_day that represents stay duration in nights |

#### price_per_night
price_per_night was derived from price divided by nights_stayed and represents an average. It is a fairer indication of customer intent. For example, a customer with an SGD 1,000 booking for 5 nights is different from a customer with an SGD 1,000 booking for 2 nights. Customers whose booking's price per night fell between SGD 375-500 and SGD 875-1000 had higher chance of not showing up.

### Categorical Data Preprocessing
While categorical values were helpful to exploratory data visualisations, they were transformed into numerical values for purposes of scaling and modelling.

| Column | Treatment |
|--|--|
| branch | Transform categorical values to binary values |
| country | Apply one-hot encoding instead of label encoding to avoid introducing ordinality between categories |
| booking_month_no | New numerical column corresponding to booking_month |

### Resampling
Data was moderately imbalanced in favour of class 0 (63% of records). If the data was not rebalanced, the resulting model may be better at predicting the majority class than minority class.

> Solution: Using resampling without replacement, the majority class was downsampled to yield 44,224 samples to match the number of minority class records, and then the majority class was upweighted by introducing a new column of weights. Newly rebalanced dataset had 88,448 records in total.

### Features and Target

- Features: branch, arrival_month_no, booking_month_no, nights_stayed, price_per_night, one-hot encoded country columns
- Target: no_show

After defining feature matrix X and target vector y, the dataset underwent a train-test split with 0.2 test size. 
> X_train.shape: (70758, 13)
> 
> y_train.shape: (70758,)
> 
> X_test.shape: (17690, 13)
> 
> y_test.shape: (17690,)

### Feature Scaling
Features contained varying scales from 10s to 1000s, and underwent Z-score normalisation to standardise their range. Statistical properties (mean and standard deviation) of X_train were used to scale both X_train and X_test.

### Modelling
Prediction of no-shows can be modelled as a probability problem (how likely would a customer no-show?) where the label follows a binomial distribution with only two outcomes, failure ('show up') or success ('no-show'). With this in mind, logistic regression is an appropriate model choice. Logistic regression, as its name implies, first performs regression, and then compresses regression outputs into probabilities using the logistic function. The model learns the best regression feature weights that maximise P(y = 1) whenever the true label y is 1 ('no-show') and maximise 1 - P(y = 1) whenever the true label y is 0 ('show up').

After the model was fitted with X_train and y_train, predictions were made on X_test. Predictions were evaluated against y_test on a set of metrics: accuracy score, F1 score and log loss, and output to `evaluation_report.txt`. 

|Metric |Result|
|--|--|
|Accuracy Score|1.0|
|F1 Score|1.0|
|Log Loss|0.00018|
