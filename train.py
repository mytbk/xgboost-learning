#!/usr/bin/env python2

import xgboost as xgb
from numpy import array
dtrain = xgb.DMatrix('out.txt.train')
dval = xgb.DMatrix('out.txt.validate')
dtest = xgb.DMatrix('out.txt.test')
param = { 'max_depth': 4,
          'eta': 0.03,
          'gamma': 1.0,
          'min_child_weight': 1,
          'save_period': 0,
          'booster': 'gbtree',
          'subsample': 0.6,
          'colsample_bytree': 0.7,
          'eval_metric': 'rmse',
          'nthreads': 4,
          'objective': 'reg:tweedie' }
num_round = 600
watchlist = [(dval, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, watchlist)
bst.dump_model('model.txt')

preds = bst.predict(dtest)
predy = []
predyint = []
outfile = open('result.txt', 'w')
for i in preds:
    predy.append(i)
    predyint.append(int(round(i)))
    outfile.write('%d\n' % int(round(i)))

outfile.close()

tf = open('out.txt.test', 'r')
testy = []
for i in tf:
    testy.append(int(i.split()[0]))

print("rmse(float): %f" % (array(predy)-array(testy)).std())
print("rmse(int): %f" % (array(predyint)-array(testy)).std())
