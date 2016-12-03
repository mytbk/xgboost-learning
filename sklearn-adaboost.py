#!/usr/bin/env python2

import convert_data
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_svmlight_file

def get_data(file):
    data = load_svmlight_file(file)
    return data[0], data[1]

train_X, train_y = get_data('out.txt.train')
test_X, test_y = get_data('out.txt.test')

rng = np.random.RandomState(1)
regr = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=2),
    n_estimators=500,
    learning_rate=0.1,
    loss='linear',
    random_state=rng)
regr.fit(train_X, train_y)
diff = regr.predict(test_X.toarray())-test_y
print('rmse = %f\n' % diff.std())
