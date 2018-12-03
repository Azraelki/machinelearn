import os
import numpy as np
import pandas as pd
import pickle
from machinelearn.chapter8.app import function4,stop
from movieclassifier.vectorizer import vect
# 序列化chapter-8中的外存学习模型
def function1():
    dest = os.path.join('movieclassifier','pkl_objects')
    if not os.path.exists(dest):
        os.makedirs(dest)
    pickle.dump(stop,open(os.path.join(dest,'stopwords.pkl'),'wb'),protocol=4)
    pickle.dump(function4(),open(os.path.join(dest,'classifier.pkl'),'wb'),protocol=4)

# 逆序列化分类器
def function2():
    clf = pickle.load(open(os.path.join("movieclassifier","pkl_objects","classifier.pkl"),'rb'))
    label = {0:'negative',1:'positive'}
    example = [" i love this movie"]
    X = vect.transform(example) # 使用映射矩阵转化样本
    print("prediction: %s\nprobability:%.2f%%"%(label[clf.predict(X)[0]],
                                                np.max(clf.predict_proba(X)*100)))



if __name__ == '__main__':
    # function1()
    function2()