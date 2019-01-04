import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

# 获取数据集
train_data = pd.read_csv("../../../machinelearndata/kaggle/digit/train.csv", encoding="utf-8")
test_data = pd.read_csv("../../../machinelearndata/kaggle/digit/test.csv", encoding="utf-8")
# train_data的数据的第一列为类标
print(train_data.shape)
print(test_data.shape)

# 特征都为0-255的数字类型
# 类标种类为0-9 one-hot化
train_y = train_data['label'].values[:,np.newaxis]
train_x = train_data.drop(columns=['label']).values

onehot = OneHotEncoder(categories='auto')
train_y = onehot.fit_transform(train_y).toarray()

# 构造一个cv
cv_split = ShuffleSplit(n_splits=10,test_size=0.3,random_state=123)
# 使用逻辑斯谛回归
logist = LogisticRegressionCV()

cv_results = cross_validate(logist,X=train_x,y=train_y,scoring='accuracy',cv=cv_split,)
print(cv_results['train_score'])
print(cv_results['test_score'])


pass