#coding:utf-8
import xgboost as xgb
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics


boston = load_boston()
# 查看波士顿数据集的keys
print(boston.keys())
boston_data=boston.data
target_var=boston.target
feature=boston.feature_names

boston_df=pd.DataFrame(boston_data,columns=boston.feature_names)
boston_df['tar_name']=target_var

# 查看目标变量描述统计
print(boston_df['tar_name'].describe())

# 把数据集转变为二分类数据
boston_df.loc[boston_df['tar_name']<=21,'tar_name']=0
boston_df.loc[boston_df['tar_name']>21,'tar_name']=1


x_train, x_test, y_train, y_test = train_test_split(boston_df[feature], boston_df['tar_name'],
                                                    test_size=0.30, random_state=1)

train_data=xgb.DMatrix(x_train,label=y_train)
dtrain=xgb.DMatrix(x_train)
dtest=xgb.DMatrix(x_test)

params={'booster':'gbtree','objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':6,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'eta': 0.03,}

watchlist = [(train_data,'train')]
bst=xgb.train(params,train_data,num_boost_round=100,evals=watchlist)


# 度量xgboost的准确性
y_train_pred = (bst.predict(dtrain)>=0.5)*1
y_test_pred =(bst.predict(dtest)>=0.5)*1
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('xgboost train/test accuracies %.3f/%.3f' % (tree_train, tree_test))


# y_pred = (ypred >= 0.5)*1
# print ('AUC: %.4f' % metrics.roc_auc_score(y_test,ypred))
# print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred))
# print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred))
# print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred))
# print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred))
# confs=metrics.confusion_matrix(y_test,y_pred)