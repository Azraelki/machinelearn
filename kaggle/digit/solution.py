import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 获取数据集
train_data = pd.read_csv("../../../machinelearndata/kaggle/digit/train.csv", encoding="utf-8")
test_data = pd.read_csv("../../../machinelearndata/kaggle/digit/test.csv", encoding="utf-8")
# train_data的数据的第一列为类标
print(train_data.shape)
print(test_data.shape)

# 特征都为0-255的数字类型
train_y = train_data['label'].values[:,np.newaxis]
train_x = train_data.drop(columns=['label']).values

# 归一化数据
std = StandardScaler()
train_x = std.fit_transform(train_x)

# 划分数据集
train_s_x,test_s_x,train_s_y,test_s_y = train_test_split(train_x,train_y,test_size=0.3,stratify=train_y)


# 使用KNN
# knc = KNeighborsClassifier(n_neighbors=10)
# knc.fit(train_s_x,train_s_y)
# print(knc.score(test_s_x,test_s_y))

# svc
svc = SVC()
svc.fit(train_s_x,train_s_y)


