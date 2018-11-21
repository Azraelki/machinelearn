import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
'''
book-name:      machine learn 
book-location:  chapter-2
content:       自适应线性神经元实现--随机梯度下降
'''
class AdalineSGD:
    def __init__(self, eta=0.01,n_iter=50,shuffle=True,random_state=None):
        '''
        :param eta: 学习速率，介于0-1之间
        :param n_iter: 迭代次数
        :param shuffle:是否重新排序样本
        :param random_state: 用于初始化权重参数的随机种子，用于复现测试结果
        '''
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self,X,y):
        # c初始化权重
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X,y = self._shuffle(X,y) # 样本重排序
            cost = []
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y) # 计算平均成本函数值
            self.cost_.append(avg_cost)
        return self

    # 不重置权重的情况下更新参数，用于在线学习，当有新的数据时可以在之前参数基础上更新
    def partial_fit(self, X,y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        return self

    # 重排序样本
    def _shuffle(self,X,y):
        r = self.rgen.permutation(len(y))
        return X[r],y[r]

    # 初始化权重
    def _initialize_weights(self,m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0,scale=0.01,size=1+m)
        self.w_initialized = True

    # 根据样本更新权重
    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta*error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:])  + self.w_[0]

    # 激活函数
    def activation(self,X):
        return X

    # 预测函数
    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)



# 标准化样本值，可以提高收敛速度和提升训练效果
def standard_X(X,y):
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
    X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()

    return X_std

def plot_decision_regions(X,y,classifier,resolution=0.02):

    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max,resolution),
                           np.arange(x2_min,x2_max,resolution))
    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],
                    y=X[y==cl,1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')








if __name__ == "__main__":
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                     header=None)

    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    ada = AdalineSGD(n_iter=15,eta=0.01,random_state=1)
    ada.fit(standard_X(X,y),y)
    # show_cost(X,y)
    plot_decision_regions(standard_X(X,y),y,classifier=ada)
    plt.title('Adaline - gradient descent')
    plt.xlabel('sepal length [standartized]')
    plt.ylabel('petal length [standartized]')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(range(1, len(ada.cost_)+1),ada.cost_,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('sum - suquared - error')
    plt.show()