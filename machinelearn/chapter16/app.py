import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pyprind
import re
import pandas as pd
from string import punctuation # 标点字符列表
from collections import Counter

# 定义一个生成器--生成mini-barches
def create_batch_generator(x,y=None,batch_size=64):
    n_batches = len(x)//batch_size
    x = x[:n_batches*batch_size]
    if y is not None:
        y = y[:n_batches*batch_size]
    for ii in range(0,len(x),batch_size):
        if y is not None:
            yield x[ii:ii+batch_size],y[ii:ii+batch_size]
        else:
            yield  x[ii:ii+batch_size]

# 导入数据
def function1():
    # 数据.csv文件从第八章中的代码中可以生成
    df = pd.read_csv('movie_data.csv',encoding='utf-8')
    counts = Counter()
    pbar = pyprind.ProgBar(len(df['review']),title='counting words occurrences')
    # 预处理数据，分割词语，计算词语的出现次数
    for i, review in enumerate(df['review']):
        text = ''.join([c if c not in punctuation else ' '+c+' ' for c in review]).lower()
        df.loc[i,'review'] = text
        pbar.update()
        counts.update(text.split())

    # 创建字符-数字映射
    word_counts = sorted(counts,key=counts.get,reverse=True)
    print(word_counts[:5])
    word_to_int = {word:ii for ii,word in enumerate(word_counts,1)}# 创建映射表

    mapped_reviews = []
    pbar = pyprind.ProgBar(len(df['review']),title='map reviews to ints')
    for review in df['review']:
        mapped_reviews.append([word_to_int[word] for word in review.split()])
        pbar.update()

    # 将映射后的数据集review固定到指定长度
    sequence_length = 200 # （同时此值也就RNN所谓的序列长度T）
    sequences = np.zeros((len(mapped_reviews),sequence_length),dtype=int)

    for i, row in enumerate(mapped_reviews):
        review_arr = np.array(row)
        sequences[i,-len(row):] = review_arr[-sequence_length:] # 低位有效，多则截取，少则左部填0

    df2 = pd.DataFrame(sequences)
    df2.to_csv('./sequences.csv')

# 整理后的数据快捷获取
def function2():
    df = pd.read_csv('movie_data.csv', encoding='utf-8')
    sequences = pd.read_csv('sequences.csv',dtype=int)
    X_train = sequences.iloc[:25000, :]
    y_train = df.loc[:25000, 'sentiment']
    X_test = sequences.iloc[25000:, :]
    y_test = df.loc[25000:, 'sentiment']

# 定义嵌入矩阵和嵌入层，将文字索引转换到一个实数区间
def function3():
    # embedding = tf.Variable(tf.random_uniform(shape=(n_words,embedding_size),minval=-1,maxval=1))
    # embed_x = tf.nn.embedding_lookup(embedding,tf_x)
    pass










if __name__ == '__main__':
    # function1()
    function2()