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


# 使用不同方式对模型进行评估
def function5():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    X = df.iloc[:,:-1].values
    y = df['MEDV'].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    slr = LinearRegression()
    slr.fit(X,y)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)

    # 评估模型性能：绘制残差图，残差=预测值-实际值
    plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',
                marker='o',edgecolor='white',label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen',
                marker='s', edgecolor='white', label='Test data')

    plt.xlabel("predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc='upper left')
    plt.hlines(y=0,xmin=0,xmax=50,color='black',lw=2)
    plt.xlim([-10,50])
    plt.show()

    # 评估模型性能：计算均方误差(MSE,既SSE的均值)，均方误差=sum（(预测值-实际值)^2）/n
    # 由输出结果得出：模型过于拟合训练数据集
    from sklearn.metrics import mean_squared_error
    print('MSE train:%.3f, test:%.3f'%(mean_squared_error(y_train,y_train_pred),
                                       mean_squared_error(y_test,y_test_pred)))

    # 评估模型性能：决定系数（R^2），R^2 = 1-(SSE/SST)，SST = sum(（真实值-均值）^2)
    from sklearn.metrics import r2_score
    print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train,y_train_pred),
                                          r2_score(y_test,y_test_pred)))

# 回归中的正则化
def function6():
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0) # ridge（岭回归），alpha为正则化强度，L2

    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=1.0) # lasso（最小绝对收缩及算子选择）,L1

    from sklearn.linear_model import ElasticNet
    elanet = ElasticNet(alpha=1.0,l1_ratio=0.5) # 弹性网络，ridge和lasso的折中，l1_ratio为L1和L2系数的比率


# 多项式回归--引入多次项拟合数据（拟合非线性关系时）
def function7():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score
    X = df[['LSTAT']].values
    y = df['MEDV'].values

    regr = LinearRegression()

    # 引入多项式
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)

    # 拟合各个多项式模型
    X_fit = np.arange(X.min(),X.max(),1)[:,np.newaxis]

    regr.fit(X,y)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y,regr.predict(X))

    regr = regr.fit(X_quad,y)
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    quadratic_r2 = r2_score(y,regr.predict(X_quad))

    regr.fit(X_cubic,y)
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    cubic_r2 = r2_score(y,regr.predict(X_cubic))

    # 画出结果
    plt.scatter(X,y,label='training points',color='lightgray')

    plt.plot(X_fit,y_lin_fit,label='linear(d=1), $R^2=%.2f'%linear_r2,
             color='blue',lw=2,linestyle=':')

    plt.plot(X_fit, y_quad_fit, label='linear(d=2), $R^2=%.2f' % quadratic_r2,
             color='red', lw=2, linestyle='-')

    plt.plot(X_fit, y_cubic_fit, label='linear(d=3), $R^2=%.2f' % cubic_r2,
             color='green', lw=2, linestyle='--')

    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('price in $1000s [MEDV]')
    plt.legend(loc='upper right')
    plt.show()


# 使用sklearn的决策树回归对非线性关系进行拟合
def function8():
    from sklearn.tree import DecisionTreeRegressor
    X = df[['LSTAT']].values
    y = df['MEDV'].values

    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X,y)
    sort_idx = X.flatten().argsort() # 绘制拟合线时按照LSTAT从小到大的顺序绘制
    lin_regplot(X[sort_idx],y[sort_idx],tree)
    # lin_regplot(X,y,tree)
    plt.xlabel("% lower status of population [LSTAT]")
    plt.ylabel("price in $1000s [MEDV]")
    plt.show()

# 使用sklearn随机森林回归对数据集进行拟合，随机森林相对决策树鲁棒性更好
def function8():
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    X = df.iloc[:,:-1].values
    y = df['MEDV'].values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=1)

    forest = RandomForestRegressor(n_estimators=1000,criterion='mse',
                                   random_state=1,n_jobs=-1)

    forest.fit(X_train,y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    print("MSE train : %.3f, test: %.3f"%(mean_squared_error(y_train,y_train_pred),
                                          mean_squared_error(y_test,y_test_pred)))

    print("R^2 train : %.3f, test: %.3f" % (r2_score(y_train, y_train_pred),
                                            r2_score(y_test,y_test_pred)))

    # 绘制残差图
    plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',
                edgecolor='white',marker='o',s=35,alpha=0.9,label='training data')
    plt.scatter(y_test_pred,y_test_pred-y_test,c='limegreen',
                edgecolor='white',marker='s',s=35,alpha=0.9,label='test data')

    plt.xlabel('predicted values')
    plt.ylabel("residuals")
    plt.legend(loc='upper left')
    plt.hlines(y=0,xmax=50,xmin=-10,lw=2,colors='black')
    plt.xlim([-10,50])
    plt.show()



if __name__ == '__main__':
    # function1()
    # function2()
    # function3()
    # function5()
    # function7()
    function8()