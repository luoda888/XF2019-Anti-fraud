# MindRank.ai

import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import scipy.spatial.distance as dist
import xgboost as xgb
import lightgbm as lgb
import catboost as cbt
import json
from sklearn.metrics import f1_score
import time
import gc
import math
from tqdm import tqdm
from scipy import stats
from sklearn.cluster import KMeans
from six.moves import reduce
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime,timedelta

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_table("../input/round1_iflyad_anticheat_traindata.txt")
test = pd.read_table("../input/round1_iflyad_anticheat_testdata_feature.txt")
all_data = train.append(test).reset_index(drop=True)

# 对时间的处理
all_data['time'] = pd.to_datetime(all_data['nginxtime']*1e+6) + timedelta(hours=8)
all_data['day'] = all_data['time'].dt.dayofyear
all_data['hour'] = all_data['time'].dt.hour

# Data Clean
# 全部变成大写，防止oppo 和 OPPO 的出现
all_data['model'].replace('PACM00',"OPPO R15",inplace=True)
all_data['model'].replace('PBAM00',"OPPO A5",inplace=True)
all_data['model'].replace('PBEM00',"OPPO R17",inplace=True)
all_data['model'].replace('PADM00',"OPPO A3",inplace=True)
all_data['model'].replace('PBBM00',"OPPO A7",inplace=True)
all_data['model'].replace('PAAM00',"OPPO R15_1",inplace=True)
all_data['model'].replace('PACT00',"OPPO R15_2",inplace=True)
all_data['model'].replace('PABT00',"OPPO A5_1",inplace=True)
all_data['model'].replace('PBCM10',"OPPO R15x",inplace=True)

for fea in ['model','make','lan']:
    all_data[fea] = all_data[fea].astype('str')
    all_data[fea] = all_data[fea].map(lambda x:x.upper())

    from urllib.parse import unquote

    def url_clean(x):
        x = unquote(x,'utf-8').replace('%2B',' ').replace('%20',' ').replace('%2F','/').replace('%3F','?').replace('%25','%').replace('%23','#').replace(".",' ').replace('??',' ').\
                               replace('%26',' ').replace("%3D",'=').replace('%22','').replace('_',' ').replace('+',' ').replace('-',' ').replace('__',' ').replace('  ',' ').replace(',',' ')
        
        if (x[0]=='V') & (x[-1]=='A'):
            return "VIVO {}".format(x)
        elif (x[0]=='P') & (x[-1]=='0'):
            return "OPPO {}".format(x)
        elif (len(x)==5) & (x[0]=='O'):
            return "Smartisan {}".format(x)
        elif ('AL00' in x):
            return "HW {}".format(x)
        else:
            return x

    all_data[fea] = all_data[fea].map(url_clean)
    
all_data['big_model'] = all_data['model'].map(lambda x:x.split(' ')[0])
all_data['model_equal_make'] = (all_data['big_model']==all_data['make']).astype(int)

# H,W,PPI

all_data['size'] = (np.sqrt(all_data['h']**2 + all_data['w'] ** 2) / 2.54) / 1000
all_data['ratio'] = all_data['h'] / all_data['w']
all_data['px'] = all_data['ppi'] * all_data['size']
all_data['mj'] = all_data['h'] * all_data['w']

num_col = ['h','w','size','mj','ratio','px']
cat_col = [i for i in all_data.select_dtypes(object).columns if i not in ['sid','label']]
both_col = []

for i in tqdm(cat_col):
    lbl = LabelEncoder()
    all_data[i+"_count"] = all_data.groupby([i])[i].transform('count')
    all_data[i+"_rank"] = all_data[i+"_count"].rank(method='min')
    all_data[i] = lbl.fit_transform(all_data[i].astype(str))
    both_col.extend([i+"_count",i+"_rank"])

for i in tqdm(['h','w','ppi','ratio']):
    all_data['{}_count'.format(i)] = all_data.groupby(['{}'.format(i)])['sid'].transform('count')
    all_data['{}_rank'.format(i)] = all_data['{}_count'.format(i)].rank(method='min')

feature_name = [i for i in all_data.columns if i not in ['sid','label','time']]
cat_list = [i for i in train.columns if i not in ['sid','label','nginxtime']]

from sklearn.metrics import roc_auc_score

tr_index = ~all_data['label'].isnull()
X_train = all_data[tr_index][list(set(feature_name))].reset_index(drop=True)
y = all_data[tr_index]['label'].reset_index(drop=True).astype(int)
X_test = all_data[~tr_index][list(set(feature_name))].reset_index(drop=True)
print(X_train.shape,X_test.shape)
random_seed = 2019
final_pred = []
cv_score = []
cv_model = []
skf = StratifiedKFold(n_splits=5, random_state=random_seed, shuffle=True)
for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
    print(index)
    train_x, test_x, train_y, test_y = X_train[feature_name].iloc[train_index], X_train[feature_name].iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    cbt_model = cbt.CatBoostClassifier(iterations=3000,learning_rate=0.05,max_depth=11,l2_leaf_reg=1,verbose=10,early_stopping_rounds=400,task_type='GPU',eval_metric='F1',cat_features=cat_list)
    cbt_model.fit(train_x[feature_name], train_y,eval_set=(test_x[feature_name],test_y))
    cv_model.append(cbt_model)
    y_test = cbt_model.predict(X_test[feature_name])
    y_val = cbt_model.predict_proba(test_x[feature_name])
    print(Counter(np.argmax(y_val,axis=1)))
    cv_score.append(f1_score(test_y,np.round(y_val[:,1])))

# Catboost比较适合类别较多的场景
    
# GPU结果五折 
# 第一折
# bestTest = 0.94051
# bestIteration = 1512

fi = []
for i in cv_model:
    tmp = {
        'name' : feature_name,
        'score' : i.feature_importances_
    }
    fi.append(pd.DataFrame(tmp))
    
fi = pd.concat(fi)
fig = plt.figure(figsize=(8,8))
fi.groupby(['name'])['score'].agg('mean').sort_values(ascending=False).head(40).plot.barh()

cv_pred = np.zeros((X_train.shape[0],))
test_pred = np.zeros((X_test.shape[0],))
for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
    print(index)
    train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    y_val = cv_model[index].predict_proba(test_x[feature_name])[:,1]
    print(y_val.shape)
    cv_pred[test_index] = y_val
    test_pred += cv_model[index].predict_proba(X_test[feature_name])[:,1] / 5

print("CV score: ",np.mean(cv_score))

submit = test[['sid']]
submit['label'] = (test_pred>=0.5).astype(int)
print(submit['label'].value_counts())
submit.to_csv("submission.csv",index=False)
