import pandas as pd
import numpy as np
from sklearn.impute import  SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from io import StringIO

# 通过pandas搜索数据缺失值
def function1():
    csv_data = \
    '''
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,
    '''
    df = pd.read_csv(StringIO(csv_data))
    print(df.isnull().sum()) # 缺失值统计
    print(df.values) # 获取数据的numpy数组形式数据
    print(df.dropna(axis=0)) # 删除缺失数据的行
    print(df.dropna(axis=1)) # 删除缺失数据的列
    print(df.dropna(how="all")) # 删除全为NaN的行
    print(df.dropna(thresh=4)) # 删除掉少于4个有效值的行
    print(df.dropna(subset=['3'])) # 删除指定列是NaN值的行

# 通过sklearn补全缺失数值
def function2():
    csv_data = \
        '''
        1.0,2.0,3.0,4.0
        5.0,6.0,,8.0
        10.0,11.0,12.0,
        '''
    df = pd.read_csv(StringIO(csv_data),header=None)

    # 通过sklearn提供的类Imputer补全缺失值
    imr = SimpleImputer(fill_value='NaN', # 缺失值标识
                  strategy='mean', # 策略，此处为均值
                  verbose=0 # 列的均值
                  )
    imr.fit(df.values)
    imputed_data = imr.transform(df.values)
    print(imputed_data)

# 处理类别数据
def function3():
    df =  pd.DataFrame([
        ['green','M',10.1,'class1'],
        ['red','L',13.5,'class2'],
        ['blue','XL',15.3,'class1']
    ])
    df.columns=['color','size','price','classlabel']
    print(df)

    # 自定义size列的值映射
    size_mapping = {'XL':3,'L':2,'M':1}
    df['size'] = df['size'].map(size_mapping)
    print(df)

    # 类标编码
    class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
    inverse_mapping = {v:k for k,v in class_mapping.items()}
    print(class_mapping)
    df['classlabel'] = df['classlabel'].map(class_mapping)
    df['classlabel'] = df['classlabel'].map(inverse_mapping)
    print(df)

    # LabelEncoder编码类标
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    print(y)
    print(class_le.inverse_transform(y))

    # 独热编码，解决标称特征的无序性，例如颜色
    X = df[['color','size','price']].values
    color_le = LabelEncoder()
    X[:,0] = color_le.fit_transform(X[:,0])
    ohe = OneHotEncoder(categorical_features=[0]) # 初始化时选定特征所在列
    print(ohe.fit_transform(X).toarray())

    print(pd.get_dummies(df[['price','color','size']]))
    print(pd.get_dummies(df[['price', 'color', 'size']],drop_first=True)) # 丢弃第一个虚拟值，可以减少变量之间的相关性(并不会造成信息的缺失)


# 划分训练和测试数据集
def function4():
    df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)

    df_wine.columns = ['Class label', 'Alcohol',
                    'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium',
                    'Total phenols', 'Flavanoids',
                    'Nonflavanoid phenols',
                    'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines',
                    'Proline']
    print('Class label',np.unique(df_wine['Class label']))
    print(df_wine.head(5)) # 输出前五行数据

    # 划分数据集
    X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    print(X_train.shape,X_test.shape)

# 特征值缩放
def function5():
    df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)

    df_wine.columns = ['Class label', 'Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # 归一化处理特征值：(x-min)/(max-min)
    mms = MinMaxScaler()
    x_train_norm = mms.fit_transform(X_train)
    x_test_norm = mms.fit_transform(X_test)

    # 标准化处理特征值：(x-均值)/标准差
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train) # 只fit一次
    X_test_std = stdsc.transform(X_test) # 此处直接转换，使用相同的拟合参数

# 正则化模型，L1正则化可以进行简单的数据稀疏处理
def function6():
    from sklearn.linear_model import LogisticRegression

    df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)

    df_wine.columns = ['Class label', 'Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


    # 标准化处理特征值：(x-均值)/标准差
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)  # 只fit一次
    X_test_std = stdsc.transform(X_test)  # 此处直接转换，使用相同的拟合参数

    # 惩罚项：penalty='l1'使用L1正则处理
    lr = LogisticRegression(penalty='l1',C=1.0)
    lr.fit(X_train_std,y_train)
    print("training accuracy: ",lr.score(X_train_std,y_train))
    print("test accuracy: ",lr.score(X_test_std,y_test))
    print(lr.intercept_) # 截距，三个值分别为当前类别相对于其他两个类别的截距
    print(lr.coef_)


# 将正则应用到多个特征上时产生的效果
def function7():
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)

    df_wine.columns = ['Class label', 'Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # 标准化处理特征值：(x-均值)/标准差
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)  # 只fit一次
    X_test_std = stdsc.transform(X_test)  # 此处直接转换，使用相同的拟合参数

    fig = plt.figure()
    ax = plt.subplot(111)

    colors = ['blue', 'green', 'red', 'cyan',
            'magenta', 'yellow', 'black',
            'pink', 'lightgreen', 'lightblue',
            'gray', 'indigo', 'orange']

    weights,params = [],[]

    for c in np.arange(-4. ,6.):
        lr = LogisticRegression(penalty='l1',C=10.0**c,random_state=0)
        lr.fit(X_train_std,y_train)
        weights.append(lr.coef_[1])
        params.append(10**c)

    weights = np.array(weights)

    for column, color in zip(range(weights.shape[1]),colors):
        plt.plot(params,weights[:,column],
                 label=df_wine.columns[column+1],
                 color=color)
    plt.axhline(0, color='black',linestyle='--',linewidth=3)
    plt.xlim([10**(-5),10**5])
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    plt.legend(loc='upper left')
    # ax.legend(loc='upper center',
    #           bbox_to_abchor=(1.38,1.03),
    #           ncol=1,fancybox=True)
    plt.show()

# 使用随机森林选择相关特征
def function8():
    from sklearn.ensemble import RandomForestClassifier

    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)

    df_wine.columns = ['Class label', 'Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    feat_labels = df_wine.columns[1:]
    forest = RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1)
    forest.fit(X_train,y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1] # 根据importances从大到小的下标排序
    for f in range(X_train.shape[1]):
        print("%2d %s %f"%(f+1,feat_labels[indices[f]],importances[indices[f]]))

    plt.title("feature importance")
    plt.bar(range(X_train.shape[1]),importances[indices],align='center')
    plt.xticks(range(X_train.shape[1]),feat_labels,rotation=90)
    plt.xlim([-1,X_train.shape[1]])
    plt.tight_layout()
    plt.show()












if __name__ == '__main__':
    # function1()
    # funtion2()
    # function3()
    # function4()
    # function5()
    # function6()
    # function7()
    function8()