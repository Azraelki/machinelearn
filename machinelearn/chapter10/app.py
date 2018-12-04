import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from LinearRegressionGD import LinearRegressionGD

df = pd.read_csv("https://raw.githubusercontent.com/rasbt/"
                 + "python-machine-learning-book-2nd-edition"
                 + "/master/code/ch10/housing.data.txt", header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


# 导入波士顿房屋数据集
def function1():

    print(df.head())

    # 绘制两两特征间的散点图（此处使用五个特征）
    sns.set(style='whitegrid',
            context='notebook')
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    sns.pairplot(df[cols],size=2.5)
    plt.show()

    # 计算五个特征的相关系数矩阵，并绘制热度图
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size':15},
                     yticklabels=cols,
                     xticklabels=cols)
    plt.show()


def lin_regplot(X,y,model):# 绘制样本散点图和模型拟合的回归线
    plt.scatter(X,y,c='steelblue',edgecolor='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None


# 拟合简单线性回归模型
def function2():
    X = df[['RM']].values
    y = df['MEDV'].values
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y[:,np.newaxis]).flatten()
    lr = LinearRegressionGD()
    lr.fit(X_std,y_std)

    sns.reset_orig() # 样式重置成matplotlib

    # 绘制迭代次数-代价函数图
    plt.plot(range(1,lr.n_iter+1),lr.cost_)
    plt.xlabel("epoch")
    plt.ylabel("SSE")
    plt.show()

    # 绘制散点图和拟合线
    lin_regplot(X_std,y_std,lr)
    plt.xlabel('average number of rooms [RM] (standardized)')
    plt.ylabel('price in $1000s [MEDV] (standardized)')
    plt.show()

# 使用sklearn估计回归模型系数
def function3():
    from sklearn.linear_model import LinearRegression

    X = df[['RM']].values
    y = df['MEDV'].values
    slr = LinearRegression()
    slr.fit(X,y)
    print('slope: %.3f' % slr.coef_[0]) # 系数
    print('intercept: %.3f' % slr.intercept_) # 截距

    # 绘制散点图和拟合线
    lin_regplot(X, y, slr)
    plt.xlabel('average number of rooms [RM] (standardized)')
    plt.ylabel('price in $1000s [MEDV] (standardized)')
    plt.show()

# 使用随机抽样一致性算法拟合模型（可以消除异常值的影响）
# 拟合出的直线由内点样本拟合，得出的系数和截距和普通的拟合不太相同
def function4():
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import LinearRegression

    X = df[['RM']].values
    y = df['MEDV'].values

    ransac = RANSACRegressor(LinearRegression(),
                             max_trials=100, # 最大迭代次数
                             min_samples=50, # 最小随机抽取样本数
                             loss='absolute_loss',# 拟合曲线和样本点的距离计算，此处为垂直绝对距离
                             residual_threshold=5.0,# 距离阈值，当在阈值内时样本加入到内点
                             random_state=0)
    ransac.fit(X,y)

    inlier_mask = ransac.inlier_mask_ # 内点集合，[true,false......]
    outlier_mask = np.logical_not(inlier_mask)
    line_X = np.arange(3,10,1)
    line_y_ransac = ransac.predict(line_X[:,np.newaxis])
    plt.scatter(X[inlier_mask],y[inlier_mask],
                c='steelblue',edgecolor='white',
                marker='o',label='Inliners')
    plt.scatter(X[outlier_mask], y[outlier_mask],
                c='limegreen', edgecolor='white',
                marker='s', label='Outliers')
    plt.plot(line_X,line_y_ransac,color='black',lw=2)

    plt.xlabel("average number of rooms [RM]")
    plt.ylabel("price in $1000s [MEDV]")
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    # function1()
    # function2()
    # function3()
    function4()