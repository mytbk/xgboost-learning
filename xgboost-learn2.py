#!/usr/bin/env python2

import sys
import xgboost as xgb
from numpy import array

def sliceidx(a,x):
    idxlow = []
    idxhigh = []
    for i in range(0, len(a)):
        if a[i]<=x:
            idxlow.append(i)
        else:
            idxhigh.append(i)
    return idxlow, idxhigh

dtrain = xgb.DMatrix('out.txt.train')
dval = xgb.DMatrix('out.txt.validate')
dtest = xgb.DMatrix('out.txt.test')

# the classification model f classifies whether y>SLICE
# if y'=(f(X)<=10) then y may <=SLICEH
SLICE = 10
SLICEL = SLICE
SLICEH = 14

### bst0 classifies whether y>SLICE ###
dtrain_lb = dtrain.get_label()
dval_lb = dval.get_label()
dtrain_blb = [int(i>SLICE) for i in dtrain_lb]
dval_blb = [int(i>SLICE) for i in dval_lb]
dtrain.set_label(dtrain_blb)
dval.set_label(dval_blb)

param = { 'max_depth': 3,
          'eta': 0.05,
          'gamma': 1.0,
          'min_child_weight': 4,
          'save_period': 0,
          'booster': 'gbtree',
          'subsample': 1,
          'colsample_bytree': 0.7,
          'eval_metric': 'error',
          'nthreads': 4,
          'objective': 'binary:logistic' }
num_round = 200
watchlist = [(dval, 'eval'), (dtrain, 'train')]
bst0 = xgb.train(param, dtrain, num_round, watchlist)

dtest_lb = dtest.get_label()[:]
pred0 = bst0.predict(dtest)

correct = 0
for i in range(0,len(dtest_lb)):
    if (pred0[i]>0.5 and dtest_lb[i]>SLICE):
        correct += 1
    elif (pred0[i]<=0.5 and dtest_lb[i]<=SLICE):
        correct += 1
    else:
        print("%d" % (dtest_lb[i]))

print("correct rate: %d/%d" % (correct, len(dtest_lb)))

dtrain.set_label(dtrain_lb)
dval.set_label(dval_lb)

### bstl do a regression for y<=SLICEH ###
idxlow, idxhigh = sliceidx(dtrain_lb,SLICEH)
dtrain_low = dtrain.slice(idxlow)

idxlow, idxhigh = sliceidx(dval_lb,SLICEH)
dval_low = dval.slice(idxlow)

param_low = { 'max_depth': 3,
              'eta': 0.03,
              'gamma': 1.0,
              'min_child_weight': 1,
              'save_period': 0,
              'booster': 'gbtree',
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'eval_metric': 'rmse',
              'nthreads': 4,
              'objective': 'count:poisson' }
num_round_low = 2000
watchlist = [(dval_low, 'evall'), (dtrain_low, 'trainl')]
bstl = xgb.train(param_low, dtrain_low, num_round_low, watchlist)
bstl.dump_model('m1l.txt')

### bsth do a regression for y>SLICEL ###
idxlow, idxhigh = sliceidx(dtrain_lb,SLICEL)
dtrain_high = dtrain.slice(idxhigh)

idxlow, idxhigh = sliceidx(dval_lb,SLICEL)
dval_high = dval.slice(idxhigh)

param_high = { 'max_depth': 4,
               'eta': 0.02,
               'gamma': 1.0,
               'min_child_weight': 2,
               'save_period': 0,
               'booster': 'gbtree',
               'subsample': 1,
               'colsample_bytree': 0.6,
               'eval_metric': 'rmse',
               'nthreads': 4,
               'objective': 'count:poisson' }
num_round_high = 1000
watchlist = [(dval_high, 'evalh'), (dtrain_high, 'trainh')]
bsth = xgb.train(param_high, dtrain_high, num_round_high, watchlist)
bsth.dump_model('m1h.txt')

### then we do a predict on the test set ###
pred0 = bst0.predict(dtest)

slice_low, slice_high = sliceidx(pred0,0.5)
dtest_low = dtest.slice(slice_low)
dtest_high = dtest.slice(slice_high)
pred_low = bstl.predict(dtest_low)
pred_high = bsth.predict(dtest_high)

predy = pred0[:]
for i in range(0,len(slice_low)):
    predy[slice_low[i]] = pred_low[i]
    pass

for i in range(0,len(slice_high)):
    predy[slice_high[i]] = pred_high[i]
    pass

predyint = [int(round(x)) for x in predy]
outfile = open('result2.txt', 'w')
for i in predyint:
    outfile.write('%d\n' % i)

outfile.close()

tf = open('out.txt.test', 'r')
testy = []
for i in tf:
    testy.append(int(i.split()[0]))

print("rmse(float): %f" % (array(predy)-array(testy)).std())
print("rmse(int): %f" % (array(predyint)-array(testy)).std())

if len(slice_low)>0:
    rmse_low = 0.0
    for i in slice_low:
        rmse_low += (predy[i]-testy[i])**2
    rmse_low = (rmse_low/len(slice_low))**0.5
    print("rmse(low): %f" % rmse_low)

if len(slice_high)>0:
    rmse_high = 0.0
    for i in slice_high:
        rmse_high += (predy[i]-testy[i])**2
    rmse_high = (rmse_high/len(slice_high))**0.5
    print("rmse(high): %f" % rmse_high)

print("low: %d    high: %d" % (len(slice_low), len(slice_high)))
