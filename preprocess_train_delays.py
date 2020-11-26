#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:07:52 2020

@author: david.hillmann
"""

import pandas as pd
import datetime
import time
import pickle


# import train ---------------------------------------------------------------

path = '/Users/david.hillmann/Documents/FullStackDS/'
train = pd.read_csv(path + "data/airline_delay_train.csv")


# preprocess train -----------------------------------------------------------

train = train.drop(columns=['FlightDate'])

# transform departure time
def transformTimes(deptime):
    t1 = deptime.apply(lambda x: time.strptime(x, '%H:%M'))
    t2 = t1.apply(lambda x: datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds())
    return t2 / 3600


train['DepTime'] = transformTimes(train.DepTime)

# one-hot encoding of categorical variables
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return res.drop(columns=[feature_to_encode])


train = encode_and_bind(train, 'Day_of_Week')
train = encode_and_bind(train, 'UniqueCarrier')
train = encode_and_bind(train, 'Origin')

def reduce_ohcols(df, regexpres, n_min):
    v1 = df.filter(regex =(regexpres)).apply(lambda x: sum(x))
    cols2drop = v1[v1 < n_min].index.values
    return df.drop(columns=cols2drop)


train = reduce_ohcols(train, 'Origin_', int(train.shape[0]*0.0025))
train = train.drop(columns=['Dest'])

X_train = train.drop(columns = 'dep_delayed_15min')
Y_train = train['dep_delayed_15min']


# hyperparameter tuning ------------------------------------------------------

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


# # Gradient Boosted Trees: Hyperparameter Search!
# gbt = GradientBoostingClassifier()

# parameters = {'max_depth':[1, 5, 10, 30], 
#               'n_estimators':[10, 20, 30],
#               'learning_rate':[0.05, 0.1, 0.2]}

# gsearch = GridSearchCV(gbt, parameters, cv = 5, scoring = 'f1', n_jobs = -1)
# gsearch.fit(X_train, Y_train)

# print(gsearch.cv_results_)
# print(gsearch.best_params_)
# print(gsearch.best_score_)


# train model ----------------------------------------------------------------

gbtmodel = GradientBoostingClassifier(n_estimators=15, 
                                      learning_rate=0.1,
                                      max_depth=30)
gbtmodel.fit(X_train, Y_train)

# save model
gbtmodel.feature_names = list(X_train.columns.values)
pickle.dump(gbtmodel, open(path + 'flights_model.sav', 'wb'))


