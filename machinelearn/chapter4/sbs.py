from sklearn.base import  clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

'''
序列特征选择算法
'''
class SBS:
    def __init__(self, estimator, k_features,scoring=accuracy_score,test_size=0.25,random_state=1):
        self.scoring = scoring # 评价模型性能方法
        self.estimator = estimator # 评估模型
        self.k_features = k_features # 最小的特征子集
        self.test_size = test_size
        self.random_state = random_state

    def fit(self,X,y):
        X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                         test_size=self.test_size,
                                                         random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim)) # 初始化获取特征下标元组
        self.subsets_ = [self.indices_] # 初始化特征下标元组列表
        score = self._calc_score(X_train,y_train,X_test,y_test,self.indices_) # 首次获取指定特征模型得分

        self.scores_ = [score] # 初始化得分列表

        while dim > self.k_features: # loop到指定最小特征数
            scores = []
            subsets = []
            for p in combinations(self.indices_,r=dim-1):
                score = self._calc_score(X_train,y_train,
                                         X_test,y_test,p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores) # 选择得分最高分值下标
            self.indices_ = subsets[best] # 获取指定特征数量的最优得分
            self.subsets_.append(self.indices_) # 收集最优得分的特征下标元组
            dim -= 1

            self.scores_.append(scores[best]) # 收集最优得分
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self,X): # 以当前特征下标元组处理数据集
        return X[:,self.indices_]
    def _calc_score(self, X_train,y_train,X_test,y_test,indices):
        self.estimator.fit(X_train[:,indices,y_train[:,indices]])
        y_pred = self.estimator.predict(X_test[:,indices])
        score = self.scoring(y_test,y_pred)
        return score


