from sklearn.datasets import load_boston
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

boston = load_boston()
#查看波士顿数据集的keys
print(boston.keys())
boston_data=boston.data
target_var=boston.target
feature=boston.feature_names

boston_df=pd.DataFrame(boston_data,columns=boston.feature_names)
boston_df['tar_name']=target_var

#查看目标变量描述统计
print(boston_df['tar_name'].describe())
#把数据集转变为二分类数据
boston_df.loc[boston_df['tar_name']<=21,'tar_name']=0
boston_df.loc[boston_df['tar_name']>21,'tar_name']=1

X_train, X_test, y_train, y_test = train_test_split(boston_df[feature], boston_df['tar_name'],
                                                    test_size=0.30, random_state=1)

GB=GradientBoostingClassifier(n_estimators=500,max_depth=2,random_state=1,learning_rate=0.03,)
GB.fit(X_train,y_train)

# 度量GBDT的准确性
y_train_pred = GB.predict(X_train)
y_test_pred = GB.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('GBDT train/test accuracies %.3f/%.3f' % (tree_train, tree_test))