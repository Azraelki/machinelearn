import numpy as np
import pandas as pd
import os
import pyprind # 进度条
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer # 考虑了词频和逆文档频率，可以降低所有样本中都包含的信息对模型的影响


# 根据原始数据构造数测试和训练数据集
def function1():
    base_path = '../../../machinelearndata/aclImdb' # 数据集解压缩后路径
    labels = {'pos':1,'neg':0}
    pbar = pyprind.ProgBar(50000)
    df = pd.DataFrame()
    for s in ('test','train'):
        for l in ('pos','neg'):
            path = os.path.join(base_path,s,l)
            for file in os.listdir(path):
                with open(os.path.join(path,file),'r',encoding='utf-8') as infile:
                    txt = infile.read()
                df = df.append([[txt,labels[l]]],ignore_index=True)
                pbar.update()
    df.columns = ['review','sentiment']

    # 将数据集重新排序并生成csv文件
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv('movie_data.csv',index=False,encoding='utf-8')

    # 测试读取csv文件
    df = pd.read_csv('movie_data.csv',encoding='utf-8')
    df.head(3)

# 清洗数据规则
def preprocessor(text):
    text = re.sub('<[^>]*>','',text) # 移除所有html标签
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text) # 寻找所有的表情符号

    # 将所有非单词类字符删除，并把表情符号追加到末尾
    text = (re.sub('[\W]+',' ', text.lower())+' '.join(emoticons).replace('-',''))
    return text

# 标记文档（将文本拆分为单独的元素，在标记的过程中一种常用的技术为词干提取技术,
# nltk中实现了许多关于自然语言处理的工具，可以使用它来对数据进行处理）
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
porter = PorterStemmer() # 词干提取器，可以将单词恢复到其原始的形式：likes -> like
# nltk.download('stopwords') # 下载停用词 例如：a, and , is ...，下载一次就可以
stop = stopwords.words('english') # 获取英文停用词

def tokenizer(text):# 不适用词干提取器
    return text.split()

def tokenizer_porter(text):# 使用词干提取器
    return [porter.stem(word) for word in text.split()]
# 预处理数据集
def function2():
    # 读取function1()生成的数据集文件
    df = pd.read_csv('movie_data.csv', encoding='utf-8')
    # 测试清洗方法是否可以正常使用
    print(preprocessor(df.loc[0,'review'][-50:]))
    print(preprocessor("</a>This :) is :( a test :-)!"))
    # 测试文档标记
    print(tokenizer("a runner likes running and runs a lot"))

    # 清晰数据集
    a = df['review']
    df['review'] = df['review'].apply(preprocessor)

    return df

# 用清洗后的文档数据集训练逻辑斯谛回归模型(全批量数据)
def function3():
    df = function2()
    X_train = df.loc[:25000,'review'].values
    y_train = df.loc[:25000,'sentiment'].values
    X_test = df.loc[25000:,'review'].values
    y_test = df.loc[25000:,'sentiment'].values

    #
    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False, # 是否转换为小写字母
                            preprocessor=None) #

    param_grid = [
        {
            'vect__ngram_range':[(1,1)],
            'vect__stop_words':[stop,None],
            'vect__tokenizer':[tokenizer,tokenizer_porter],
            'clf__penalty':['l1','l2'],
            'clf__C':[1.0,10.0,100.0]
        },
        {
            'vect__ngram_range': [(1, 1)],
            'vect__stop_words': [stop, None],
            'vect__tokenizer': [tokenizer, tokenizer_porter],
            'vect__use_idf':[False], # 不适用idf
            'vec__norm':[False],
            'clf__penalty': ['l1', 'l2'],
            'clf__C': [1.0, 10.0, 100.0]
        }
    ]

    lr_tfidf = Pipeline([('vect',tfidf),
                        ('clf',LogisticRegression(random_state=0))])

    gs_lr_tfidf = GridSearchCV(lr_tfidf,param_grid,
                               scoring='accuracy',cv=5,
                               verbose=1, # 控制信息详细程度
                               n_jobs=1)
    gs_lr_tfidf.fit(X_train,y_train)

    print('best parameter set:%s' % gs_lr_tfidf.best_params_)

    print('cv accuracy: %.3f' % gs_lr_tfidf.best_score_)

    clf = gs_lr_tfidf.best_estimator_
    print('test accuracy: %.3f'%clf.score(X_test,y_test))
###############################################################################
# 用清洗后的文档数据集训练逻辑斯谛回归模型（在线算法和外村学习）
def tokenizer_stop_word(text):# 使用停用字的数据清洗方法
    text = re.sub('<[^>]*>', '', text)  # 移除所有html标签
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())  # 寻找所有的表情符号

    # 将所有非单词类字符删除，并把表情符号追加到末尾
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))

    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# 文档生成器
def stream_docs(path):
    with open(path,'r',encoding='utf-8') as csv:
        next(csv) # 跳过header
        for line in csv:
            text,label = line[:-3],int(line[-2])
            yield text,label

# 批量获取样本数据
def get_minibatch(doc_stream,size):
    '''
    :param doc_stream: 数据生成器对象
    :param size: 获取的数量
    :return: 样本列表，label列表
    '''
    docs,y = [],[]
    try:
        for _ in range(size):
            text,label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None,None
    return docs,y

def function4():
    # 由于无法将数据集一次存入内存，所有无法使用CountVectorizer生成词频，也无法使用
    # TfidfVectorizer计算逆文档频率，这里使用HashingVectorizer做相似的事
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier # 随机梯度下降
    # 测试文档生成器
    # print(next(stream_docs(path='movie_data.csv')))

    vect = HashingVectorizer(decode_error='ignore',
                             n_features=2**21, # 特征列数量
                             preprocessor=None, # 重新预处理
                             tokenizer=tokenizer_stop_word)
    clf = SGDClassifier(loss='log',# 损失函数，这里log指使用逻辑斯谛回归分类器
                        random_state=1,
                        n_iter=1)

    doc_stream = stream_docs(path='movie_data.csv')
    pbar = pyprind.ProgBar(45)
    classes = np.array([0,1])
    for _ in range(45):# 使用45000个样本进行训练模型
        X_train,y_train = get_minibatch(doc_stream,size=1000)
        if not X_train:
            return
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train,y_train,classes=classes)
        pbar.update()

    # 使用最后5000组数据进行评估性能
    X_test,y_test = get_minibatch(doc_stream,size=5000)
    X_test = vect.transform(X_test)
    print("accuracy: %.3f" % clf.score(X_test,y_test))
    return clf


# 以上的文档分解没有考虑上句子的结构和语法，另一种文档分解的算法为LDA（狄利克雷分类）
def function5():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    df = pd.read_csv("movie_data.csv",encoding='utf-8')
    count = CountVectorizer(stop_words='english',max_df=0.1,max_features=5000)

    X = count.fit_transform(df['review'].values)
    lda = LatentDirichletAllocation(n_topics=10, # 分成的主题数量
                                    random_state=123,
                                    learning_method='batch')
    X_topics = lda.fit_transform(X)
    print(lda.components_.shape) # 10*5000,十个主题的对应5000词汇递增列表重要性矩阵

    # 打印十个主题前五重要性的词汇
    n_top_words = 5
    feature_names = count.get_feature_names()
    for topic_idx,topic in enumerate(lda.components_):
        print("topic %d " % (topic_idx+1))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1] ]))





if __name__ == '__main__':
    # function1()
    # function2()
    # function3()
    # function4()
    function5()




