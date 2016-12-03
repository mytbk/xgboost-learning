#!/usr/bin/env python2

import xgboost as xgb
dtrain = xgb.DMatrix('out.txt.train')
dval = xgb.DMatrix('out.txt.validate')
dtest = xgb.DMatrix('out.txt.test')
param = { 'max_depth': 3,
          'eta': 0.05,
          'gamma': 1.0,
          'min_child_weight': 1,
          'save_period': 0,
          'booster': 'gbtree',
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'eval_metric': 'rmse',
          'nthreads': 2,
          'objective': 'reg:tweedie' }
num_round = 500
watchlist = [(dval, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, watchlist)
bst.dump_model('model.txt')

preds = bst.predict(dtest)
predy = []
outfile = open('result.txt', 'w')
for i in preds:
    predy.append(i)
    outfile.write('%f\n' % i)

outfile.close()

tf = open('out.txt.test', 'r')
testy = []
for i in tf:
    testy.append(int(i.split()[0]))

print('%d %d\n'%(len(predy),len(testy)))
rmse = 0.0
for i in range(0, len(predy)):
    rmse += (predy[i]-testy[i])**2

rmse = (rmse/len(predy))**0.5
print('rmse=%f'%rmse)
