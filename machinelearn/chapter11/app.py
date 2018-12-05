import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 生成并绘制随机散点图,之后使用sklearn的k-mean模型聚类数据
def function1():
    X,y = make_blobs(n_samples=150,
                     n_features=2,
                     centers=3,
                     cluster_std=0.5,
                     shuffle=True,
                     random_state=0)
    plt.scatter(X[:,0],X[:,1],c='white',marker='o',edgecolor='black',s=50)
    plt.grid()
    plt.show()

    # 使用sklearn提供的k-mean模型聚类
    km = KMeans(n_clusters=3, # 簇数
                init='random',# 初始点的初始方式
                n_init=10, # 初始化n_init次初始点，并独立运行n_init轮聚聚类，之后选择最优模型
                max_iter=300, # 每轮寻找簇所迭代的次数
                tol=1e-4, # 簇内平方和误差容忍度
                random_state=0)
    y_km = km.fit_predict(X)

    # 绘制聚类结果
    plt.scatter(X[y_km==0,0],X[y_km==0,1],
                s=50,c='lightgreen',marker='s',edgecolors='black',
                label='cluster1')
    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
                s=50, c='blue', marker='o', edgecolors='black',
                label='cluster2')
    plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1],
                s=50, c='green', marker='x', edgecolors='black',
                label='cluster3')
    # 绘制中心点
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],
            s=250,marker='*',c='red',edgecolor='black',label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()

# 使用k-mean++算法解决k-mean算法随机初始化中心点时可能导致的收敛效果不佳和收敛速度慢
# 的情况，
def function2():
    X, y = make_blobs(n_samples=150,
                      n_features=2,
                      centers=3,
                      cluster_std=0.5,
                      shuffle=True,
                      random_state=0)
    plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50)
    plt.grid()
    plt.show()

    # 使用sklearn提供的k-mean模型聚类
    km = KMeans(n_clusters=3,  # 簇数
                # init='random',  # 初始点的初始方式，默认使用k-means++方式
                n_init=10,  # 初始化n_init次初始点，并独立运行n_init轮聚聚类，之后选择最优模型
                max_iter=300,  # 每轮寻找簇所迭代的次数
                tol=1e-4,  # 簇内平方和误差容忍度
                random_state=0)
    y_km = km.fit_predict(X)

    # 绘制聚类结果
    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
                s=50, c='lightgreen', marker='s', edgecolors='black',
                label='cluster1')
    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
                s=50, c='blue', marker='o', edgecolors='black',
                label='cluster2')
    plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1],
                s=50, c='green', marker='x', edgecolors='black',
                label='cluster3')
    # 绘制中心点
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                s=250, marker='*', c='red', edgecolor='black', label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()

# 通过肘方法评估聚类效果
def function3():
    X, y = make_blobs(n_samples=150,
                      n_features=2,
                      centers=3,
                      cluster_std=0.5,
                      shuffle=True,
                      random_state=0)
    plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50)
    plt.grid()
    plt.show()

    # 使用sklearn提供的k-mean模型聚类
    km = KMeans(n_clusters=3,  # 簇数
                # init='random',  # 初始点的初始方式，默认使用k-means++方式
                n_init=10,  # 初始化n_init次初始点，并独立运行n_init轮聚聚类，之后选择最优模型
                max_iter=300,  # 每轮寻找簇所迭代的次数
                tol=1e-4,  # 簇内平方和误差容忍度
                random_state=0)
    y_km = km.fit_predict(X)

    # 聚类偏差（簇内误差平方和）
    print("Distortion: %.2f" % km.inertia_)

    # 绘制不同k值时聚类偏差的变化，选取转折点的K值为簇数量
    distortions = []
    for i in range(1,11):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)

    plt.plot(range(1,11),distortions,marker='o')
    plt.xlabel("number of clusters")
    plt.ylabel("distortion")
    plt.show()

# 通过轮廓图评价聚类效果
def function4():
    from sklearn.metrics import silhouette_samples
    X, y = make_blobs(n_samples=150,
                      n_features=2,
                      centers=3,
                      cluster_std=0.5,
                      shuffle=True,
                      random_state=0)
    plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50)
    plt.grid()
    plt.show()

    # 使用sklearn提供的k-mean模型聚类
    km = KMeans(n_clusters=3,  # 簇数
                # init='random',  # 初始点的初始方式，默认使用k-means++方式
                n_init=10,  # 初始化n_init次初始点，并独立运行n_init轮聚聚类，之后选择最优模型
                max_iter=300,  # 每轮寻找簇所迭代的次数
                tol=1e-4,  # 簇内平方和误差容忍度
                random_state=0)
    y_km = km.fit_predict(X)

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    # 计算轮廓系数
    silhosuette_vals = silhouette_samples(X,y_km,metric='euclidean')

    y_ax_lower,y_ax_upper = 0,0
    yticks = []
    # 绘制轮廓图
    for i,c in enumerate(cluster_labels):
        c_silhouette_vals = silhosuette_vals[y_km==c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i)/n_clusters)
        plt.barh(range(y_ax_lower,y_ax_upper),c_silhouette_vals,
                 height=1.0,
                 edgecolor='none',
                 color=color)
        yticks.append((y_ax_lower+y_ax_upper)/2.)
        y_ax_lower += len(c_silhouette_vals)
    silhosuette_avg = np.mean(silhosuette_vals)
    plt.axvline(silhosuette_avg,color='red',linestyle='--')
    plt.yticks(yticks,cluster_labels+1)
    plt.xlabel("silhouette coefficient")
    plt.ylabel("cluster")
    plt.show()




if __name__ == '__main__':
    # function1()
    # function2()
    # function3()
    function4()