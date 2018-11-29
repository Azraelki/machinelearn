import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 绘制决策边界
def plot_decision_regions(X,y,classifier,resolution=0.02):
    from matplotlib.colors import ListedColormap
    markers = ['s','x','o','^','v']
    colors = ['red','blue','lightgreen','gray','cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1,X[:,0].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                          np.arange(x2_min,x2_max,resolution))

    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx , cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],
                    y=X[y==cl,1],
                    c=cmap(idx),
                    edgecolors='black',
                    marker=markers[idx],
                    label=cl)

################################### 无监督数据降维计数--主成分分析###########################################
# 自己实现的PCA:   用numpy计算样本的协方差矩阵的特征对，并选择映射矩阵将特征映射到低维子空间
def function1(X_train_std):
    df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)

    cov_mat = np.cov(X_train_std.T) # 获取样本协方差矩阵
    eigen_vals,eigen_vecs = np.linalg.eig(cov_mat) # 根据协方差矩阵获取特征值和对应的特征向量
    print(" eigenvalues \n%s" % eigen_vals)
    print(eigen_vecs)

    # 绘制特征值的方差贡献率（特征值/sum(特征值)）
    tot = sum(eigen_vals)
    var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)] # 从大到小排列

    cum_var_exp = np.cumsum(var_exp) # 获取累加的列表
    plt.bar(range(1,14),var_exp,alpha=0.5,align='center',label='individual explained variance') # 柱形图
    plt.step(range(1,14),cum_var_exp,where='mid',label='cumulative explained variance')
    plt.ylabel("explained variance ratio")
    plt.xlabel("principal component index")
    plt.legend(loc='best')
    plt.show()

    # 构造映射矩阵并将特征映射到选定的低维子空间上
    eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(key=lambda k:k[0],reverse=True) # 降序排列

    # 根据前两个特征生成13*2的映射矩阵（前两个为影响最大的特征值）
    w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
    print('matrix W:\n',w)

    # 使用映射矩阵转换样本
    X_train_pca = X_train_std.dot(w)

    colors = ['r','b','g']
    markers = ['s','x','o']
    for l,c,m in zip(np.unique(y_train),colors,markers):
        plt.scatter(X_train_pca[y_train==l,0],
                    X_train_pca[y_train==l,1],
                    c=c,label=l,marker=m)

    plt.xlabel('pc 1')
    plt.ylabel('pc 2')
    plt.legend(loc='lower left')
    plt.show()


# 使用sklearn的PCA进行数据预处理，使用逻辑斯谛回归模型进行分类，并绘制决策边界图
def function2():
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA

    df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)

    pca = PCA(n_components=2) # 创建PCA类，当n_components=None时保留所有的主成分
    lr = LogisticRegression() # 创建逻辑斯谛回归分类器
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # print(pca.explained_variance_ratio_) # 不同特征的方差贡献率降序列表（n_components=None时生效）

    lr.fit(X_train_pca,y_train)
    plot_decision_regions(X_train_pca,y_train,classifier=lr) # 绘制训练数据决策边界
    plt.xlabel('pc 1')
    plt.ylabel('pc 2')
    plt.legend(loc='lower left')
    plt.show()

    plot_decision_regions(X_test_pca,y_test,classifier=lr) # 绘制测试数据决策边界
    plt.xlabel('pc 1')
    plt.ylabel('pc 2')
    plt.legend(loc='lower left')
    plt.show()


################################### 监督降维技术--线性判别分析###########################################

# 手工实现LDA（线性判别分析）：极速昂标准化之后的样本的类内散布矩阵、类间散布矩阵、全局散布矩阵，然后计算特征值和特征向量，并映射到信息保留最大的特征上
def function3():
    df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)

    np.set_printoptions(precision=4) # 设置numpy打印精度，默认为8位

    # 计算各类别特征均值向量
    mean_vecs = []
    for label in range(1,4):
        mean_vecs.append(np.mean(X_train_std[y_train==label],axis=0))
        print('MV %s: %s\n' % (label,mean_vecs[label-1]))

    # 累加各类别的散布矩阵（方法一）-----类内
    d = 13 # 特征数量
    S_W = np.zeros((d,d)) #累加后的散布矩阵初始化
    for label,mv in zip(range(1,4),mean_vecs):
        class_scatter = np.zeros((d,d))
        for row in X_train_std[y_train==label]:
            row,mv = row.reshape(d,1),mv.reshape(d,1)
            class_scatter += (row-mv).dot((row-mv).T)
        S_W += class_scatter
    print("Widthin-class scatter matrix: %sx%s" %( S_W.shape[0],S_W.shape[1]))

    # 累加各类别的散布矩阵（方法二）-----类内
    # 由上面可以看到，各类别 散布矩阵/类别样本数时 计算散布矩阵的计算方式和计算协方差的方式一样，
    d = 13
    S_W = np.zeros((d,d))
    for label,mv in zip(range(1,4),mv):
        class_scatter = np.cov(X_train_std[y_train==label].T)
        S_W += class_scatter
    print("scaled within-class scatter matrix: %sx%s" % (S_W.shape[0],S_W.shape[1]))

    # 全类别样本的散布矩阵 --------类间
    mean_overall = np.mean(X_train_std,axis=0)
    d = 13
    S_B = np.zeros((d,d))
    for i ,mean_vec in enumerate(mean_vecs):
        n = X_train[y_train == i+1,:].shape[0] # 样本i的个数
        mean_vec = mean_vec.reshape(d,1)
        mean_overall = mean_overall.reshape(d,1)
        S_B += n*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)

    print("between-class scatter matrix: %sx%s"%(S_B.shape[0],S_B.shape[1]))

    # 获取特征值和特征向量
    eigen_vals,eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # 构造特征值和特征向量元组列表
    eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    # 降序排列
    eigen_pairs = sorted(eigen_pairs,key=lambda k:k[0],reverse=True)

    print("eigencalues in descending order:\n")
    for eigen_val in eigen_pairs:
        print(eigen_val[0])

    # 绘制特征对线性判别信息保留程度的降序图
    tot = sum(eigen_vals.real)
    discr = [(i/tot) for i in sorted(eigen_vals.real,reverse=True)]
    cum_discr = np.cumsum(discr)
    plt.bar(range(1,14),discr,alpha=0.5,align='center',label='individual "discriminability"')
    plt.step(range(1,14),cum_discr,where='mid',label='cumulative "discriminablity"')
    plt.ylabel('"discriminabilty" ratio')
    plt.xlabel('Liner discriminants')
    plt.legend(loc='best')
    plt.show()

    # 叠加判别能力最强的前两个特征向量
    w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
    print("Matrix W :\n",w) # d*2矩阵

    # 将样本映射到新的特征空间上
    X_train_lda = X_train_std.dot(w)
    colors = ['r','b','g']
    makers = ['s','x','o']
    for l,c,m in zip(np.unique(y_train),colors,makers):
        plt.scatter(X_train_lda[y_train==l,0],
                    X_train_lda[y_train==l,1]*(-1),
                    c=c,label=l,marker=m)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower right')
    plt.show()

# sklearn方式实现LDA
def function4():
    from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis as LDA
    from sklearn.linear_model import LogisticRegression

    df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)

    np.set_printoptions(precision=4)  # 设置numpy打印精度，默认为8位

    lda = LDA(n_components=2)
    # 训练数据决策边界
    X_train_lda = lda.fit_transform(X_train_std,y_train)
    lr = LogisticRegression()
    lr.fit(X_train_lda,y_train)
    plot_decision_regions(X_train_lda,y_train,classifier=lr)
    plt.xlabel("LD 1")
    plt.ylabel("LD 2")
    plt.legend(loc='lower left')
    plt.show()

    # 测试数据决策边界
    X_test_lda = lda.transform(X_test_std)
    plot_decision_regions(X_test_lda, y_test, classifier=lr)
    plt.xlabel("LD 1")
    plt.ylabel("LD 2")
    plt.legend(loc='lower left')
    plt.show()












if __name__ == '__main__':
    # function1()
    # function2()
    function4()




