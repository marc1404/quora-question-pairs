import xgboost as xgb

dtrain = xgb.DMatrix('data/train_vector.csv')
dtest = xgb.DMatrix('data/test_vector.csv')

param = {
    'max_depth': 2,
    'eta': 1,
    'silent': 1,
    'objective': 'binary:logistic'
}

num_round = 2

boost = xgb.train(param, dtrain, num_round)
predictions = boost.predict(dtest)
