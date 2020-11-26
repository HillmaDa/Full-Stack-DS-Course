#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:07:52 2020

@author: david.hillmann
"""


# import & preprocess scoring data and trained model --------------------------

import pandas as pd
import datetime
import time
import pickle

path = '/Users/david.hillmann/Documents/FullStackDS/'
topredict_df = pd.read_csv(path + "data/airline_delay_test.csv")
scoring_df = topredict_df.drop(columns=['FlightDate'])

# import trained model
gbtmodel = pickle.load(open(path + 'flights_model.sav', 'rb'))

# transform departure time
def transformTimes(deptime):
    t1 = deptime.apply(lambda x: time.strptime(x, '%H:%M'))
    t2 = t1.apply(lambda x: datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds())
    return t2 / 3600

scoring_df['DepTime'] = transformTimes(scoring_df.DepTime)

# one-hot encoding of categorical variables
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return res.drop(columns=[feature_to_encode])


scoring_df = encode_and_bind(scoring_df, 'Day_of_Week')
scoring_df = encode_and_bind(scoring_df, 'UniqueCarrier')
scoring_df = encode_and_bind(scoring_df, 'Origin')
scoring_df = scoring_df.drop(columns=['Dest'])

# if label known
X_scoring = scoring_df.drop(columns = 'dep_delayed_15min')
Y_scoring = scoring_df['dep_delayed_15min']

# if label unknown
# X_scoring = scoring_df

# Get missing columns in the training scoring_df
missing_cols = set(gbtmodel.feature_names) - set(X_scoring.columns)

# Add a missing column in scoring_df set with default value equal to 0
for c in missing_cols:
    X_scoring[c] = 0

# Ensure the order of column in the scoring_df set is in the same order than in train set
X_scoring = X_scoring[gbtmodel.feature_names]



# predict & export -----------------------------------------------------------

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score

# # metrics: precision & recall!
# print(precision_recall_fscore_support(Y_scoring, gbtmodel.predict(X_scoring)))

# # mcc [-1;+1] Werte um Null ~ Zufall
# print(matthews_corrcoef(Y_scoring, gbtmodel.predict(X_scoring)))

# # Area Under the Curve (AUC) der ROC Kurve (0.5 ~ Zufall)
# print(roc_auc_score(Y_scoring, gbtmodel.predict_proba(X_scoring)[:,1]))

# # Kreuztabelle
# print(pd.crosstab(gbtmodel.predict(X_scoring), Y_scoring))

# export predictions
scores = gbtmodel.predict_proba(X_scoring)[:,1]
topredict_df['delay_pred_prob'] = scores
topredict_df['delay_pred_bin'] = scores >= 0.5

topredict_df.to_csv(path + 'predictions/predicted_delays.csv')



