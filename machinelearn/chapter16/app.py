import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pyprind
import re
import pandas as pd
from string import punctuation # 标点字符列表
from collections import Counter
from SentimentRNN import SentimentRNN
from CharRNN import CharRnn

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

#############################RNN-1 电影情感分析#########################################
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

# 测试创建的模型--情感分析
def function4():
    df = pd.read_csv('movie_data.csv', encoding='utf-8')
    sequences = pd.read_csv('sequences.csv', dtype=int)
    sequences = sequences.iloc[:,1:]
    X_train = sequences.iloc[:25000, :]
    y_train = df.loc[:25000, 'sentiment']
    X_test = sequences.iloc[25000:, :]
    y_test = df.loc[25000:, 'sentiment']

    n_words = np.max(np.array(sequences.iloc[:,:]).flatten())
    rnn = SentimentRNN(n_words=n_words,
                       seq_len=200,
                       embed_size=256,
                       lstm_size=128,
                       num_layers=1,
                       batch_size=100,
                       learning_rate=0.001)

    # rnn.train(X_train,y_train,num_epochs=40)
    preds = rnn.predict(X_test)
    y_true = y_test[:len(preds)]
    print(" test acc : %.3f"%(np.sum(preds==y_true)/len(y_true)))

#############################RNN-2 字符预测#########################################
def reshape_data(sequence,batch_size,num_steps):
    ''' 构造输入数据的形状
    :param sequence: 转化为int列表的数据集列表
    :param batch_size: 批次数
    :param num_steps: 步长，每步所包含的字符的数量
    :return:
    '''
    tot_batch_length = batch_size * num_steps # 每个批次包含的字符长度
    num_batches = int(len(sequence)/tot_batch_length) # 批次数量

    if num_batches * tot_batch_length + 1 > len(sequence):
        num_batches = num_batches - 1 # 保证每批次的字符数量是满的
    # 截断不满一批次的数据
    x = sequence[0:num_batches*tot_batch_length]
    y = sequence[1:num_batches*tot_batch_length+1]

    # 分拆为批次数据
    x_batch_splits = np.split(x,batch_size)
    y_batch_splits = np.split(y,batch_size)

    # 把批次数据堆到一起
    # batch_size * tot_batch_length
    x = np.stack(x_batch_splits)
    y = np.stack(y_batch_splits)

    return x,y

def create_batch_generator(data_x,data_y,num_steps): # 小批次生成器，每次弹出num_steps长度的数据
    batch_size , tot_batch_length = data_x.shape
    num_batches = int(tot_batch_length/num_steps)
    for b in range(num_batches):
        yield (data_x[:,b*num_steps:(b+1)*num_steps],
               data_y[:,b*num_steps:(b+1)*num_steps])

# 准备数据，一篇戏剧
def function5():
    # 读取数据
    with open('pg2265.txt','r',encoding='utf-8') as f:
        text = f.read()
    text = text[15858:]
    chars = set(text)
    char2int = {ch:i for i,ch in enumerate(chars)}
    int2char = dict(enumerate(chars))
    text_ints = np.array([char2int[ch] for ch in text],dtype=np.int32) # 将text转化为int

    # 训练模型
    batch_size = 64
    num_steps = 100
    train_x,train_y = reshape_data(text_ints,batch_size,num_steps)

    rnn = CharRnn(num_classes=len(chars),batch_size=batch_size,
                  char2int=char2int,int2char=int2char,chars=chars)

    rnn.train(train_x,train_y,num_epochs=100,ckpt_dir='./model-100/')

    del rnn
    np.random.seed(123)
    rnn = CharRnn(len(chars),sampling=True,int2char=int2char,char2int=char2int,chars=chars)
    print(rnn.sample(ckpt_dir='./model-100/',output_length=500))












if __name__ == '__main__':
    # function1()
    # function2()
    # function4()
    function5()