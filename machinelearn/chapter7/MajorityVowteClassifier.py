from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajoritVoteClassifier(BaseEstimator,ClassifierMixin):
    '''
    多数投票集合分类器
    '''
    def __init__(self,classifiers,vote='classlabel',weights=None):
        '''
        :param classifiers: 分类器列表
        :param vote: 投票依据类标还是概率，可选值{'classlabel','probability'}
        :param weights: 分类器的权重
        '''
        self.classifiers = classifiers
        self.named_classifiers = {key:value for key,value in _name_estimators(classifiers)} # 分类器命名
        self.vote = vote
        self.weights = weights

    def fit(self,X,y):
        '''
        训练分类器
        :param X: 样本矩阵
        :param y: 类别矩阵
        :return: self
        '''
        self.lablenc_ = LabelEncoder() # 初始化类标解析器
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_ # 保存类别信息
        self.classifiers_ = [] # 存放训练过后的评估器

        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X,self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self,X):
        '''
        预测X样本的类标
        :param X: 样本
        :return: 投票后预测的类标
        '''
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X),axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x:np.argmax(np.bincount(x,weights=self.weights)),
                                           axis=1,arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self,X):
        '''
        预测类别概率
        :param X: 样本
        :return: 各类别概率
        '''
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])

        avg_proba = np.average(probas,axis=0,weights=self.weights)
        return avg_proba

    def get_params(self,deep=True):
        '''
        获取gridsearch分类器参数名
        :param deep:
        :return:
        '''
        if not deep:
            return super(MajoritVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):# 此处使用six是为了兼容python2.7
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name,key)] = value

            return out

