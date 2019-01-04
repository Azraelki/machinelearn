## 算法模型
from sklearn import svm,tree,linear_model,neighbors,naive_bayes,ensemble,discriminant_analysis
from sklearn import gaussian_process
from xgboost import XGBClassifier

## 模型处理工具
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import metrics
import pandas as pd

## 可视化工具
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns
import graphviz
# from pandas.tools.plotting import scatter_matrix

import time

# 设置图形展示风格
mpl.style.use('ggplot')
sns.set_style('white')
plb.rcParams['figure.figsize'] = 12,8
is_show = False
# 获取数据集
data_raw  = pd.read_csv("../../../machinelearndata/kaggle/titanic/train.csv", encoding="utf-8")
data_val   = pd.read_csv("../../../machinelearndata/kaggle/titanic/test.csv", encoding="utf-8")

data1 = data_raw.copy(deep=True)
data_cleaner = [data1,data_val]

# 展示数据
print(data_raw.info())
print(data_raw.sample(10))

print('训练列数据的缺失值：\n',data1.isnull().sum())
print('-'*10)

print('测试列数据缺失值：\n',data_val.isnull().sum())
print('-'*10)
print(data_raw.describe(include='all'))

# 1、补充或者删除两个数据集的缺失值

for dataset in data_cleaner:
    # 使用中位数数补全Age
    dataset['Age'].fillna(dataset['Age'].median(),inplace=True)
    # 使用众数补全Embarked
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)
    # 使用中位数补全Fare
    dataset['Fare'].fillna(dataset['Fare'].median(),inplace=True)

# 删除列
drop_column = ['PassengerId','Cabin','Ticket']
data1.drop(drop_column,axis=1,inplace=True)

print(data1.isnull().sum())
print('-'*10)
print(data_val.isnull().sum())

# 2、特征工程
for dataset in data_cleaner:
    # 创建家庭字段
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

    # 获取名字的title
    dataset['Title'] = dataset['Name'].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]

    # 将Fare和Age分段
    dataset['FareBin'] = pd.qcut(dataset['Fare'],4)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int),5)

# 清理稀有的title
print(data1['Title'].value_counts())
stat_min = 10 # 出现最小次数
title_names = (data1['Title'].value_counts()<stat_min)
data1['Title'] = data1['Title'].apply(lambda x:'Misc' if title_names.loc[x] else x)
print(data1['Title'].value_counts())
print("-"*10)

print(data1.info())
print(data_val.info())
print(data1.sample(10))

##3、 转换格式
#
label = LabelEncoder()
for dataset in data_cleaner:
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

# 定义 y 量
Target = ['Survived']
# 定义x变量
data1_x = ['Sex','Pclass','Embarked','Title','SibSp','Parch','Age','Fare',
           'FamilySize','IsAlone']
data1_x_clac = ['Sex_Code','Pclass','Embarked_Code','Title_Code','SibSp','Parch',
                'Age','Fare']

data1_xy = Target+data1_x
print('original x y:',data1_xy,'\n')

data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('bin x y',data1_xy_bin,'\n')

data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('dummy x y:',data1_xy_dummy,'\n')

a = data1_dummy.head()

# 再次检验清洗后的数据
print('train columns with null values:\n',data1.isnull().sum())
print('-'*10)
print(data1.info())
print('-'*10)

print('test columns with null values:\n',data_val.isnull().sum())
print('-'*10)
print(data_val.info())
print('-'*10)

print(data_raw.describe(include='all'))

## 3.1、划分训练集和测试集

train1_x,test1_x,train1_y,test1_y = model_selection.train_test_split(data1[data1_x_clac],data1[Target],random_state=0)
train1_x_bin,test1_x_bin,train1_y_bin,test1_y_bin = model_selection.train_test_split(data1[data1_x_bin],data1[Target],random_state=0)
train1_x_dummy,test1_x_dummy,train1_y_dummy,test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy],data1[Target],random_state=0)

print('data1 shape:{}'.format(data1.shape))
print('train1 shape:{}'.format(train1_x.shape))
print('test1 shape:{}'.format(test1_x.shape))

a = train1_x_bin.head()


## 4、分析数据

# 描述变量和Survived字段的关系
for x in data1_x:
    if data1[x].dtype != 'float64':
        print('survived correlation by:',x)
        print(data1[[x,Target[0]]].groupby(x,as_index=False).mean())
        print('-'*10,'\n')

# 图形分析
def graph_show():
    # 图形描述
    plt.figure(figsize=[16,12])
    plt.subplot(231)
    plt.boxplot(x=data1['Fare'],showmeans=True,meanline=True)
    plt.title('Fare Boxplot')
    plt.ylabel('fare ($)')

    plt.subplot(232)
    plt.boxplot(data1['Age'],showmeans=True,meanline=True)
    plt.title('age boxplot')
    plt.ylabel('age (years)')

    plt.subplot(233)
    plt.boxplot(data1['FamilySize'],showmeans=True,meanline=True)
    plt.title('Family size boxplot')
    plt.ylabel('family size (#)')

    plt.subplot(234)
    plt.hist(x=[data1[data1['Survived']==1]['Fare'],data1[data1['Survived']==0]['Fare']],
             stacked=True,color=['g','r'],label=['survived','dead'])
    plt.title('Fare histogram by survival')
    plt.xlabel('fare ($)')
    plt.ylabel('# of passengers')
    plt.legend()

    plt.subplot(235)
    plt.hist(x=[data1[data1['Survived']==1]['Age'],data1[data1['Survived']==0]['Age']],
             stacked=True,color=['g','r'],label=['survived','dead'])
    plt.title('age histogram by survival')
    plt.xlabel('age (years)')
    plt.ylabel('# of passengers')
    plt.legend()

    plt.subplot(236)
    plt.hist(x=[data1[data1['Survived']==1]['FamilySize'],data1[data1['Survived']==0]['FamilySize']],
             stacked=True,color=['g','r'],label=['survived','dead'])
    plt.title('family size histogram by survival')
    plt.xlabel('family size (#)')
    plt.ylabel('# of passengers')
    plt.legend()
    if is_show:
        plt.show()

    # 使用seaborn描绘多变量
    fig,saxis = plt.subplots(2,3,figsize=(16,12))
    sns.barplot(x='Embarked',y='Survived',data=data1,ax=saxis[0,0])
    sns.barplot(x='Pclass',y='Survived',order=(1,2,3),data=data1,ax=saxis[0,1])
    sns.barplot(x='IsAlone',y='Survived',order=(1,0),data=data1,ax=saxis[0,2])

    sns.pointplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])
    sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])
    sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])

    if is_show:
        plt.show()

    # 比较Pclass和其他特征
    fig,(axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))
    sns.boxplot(x='Pclass',y='Fare',hue='Survived',data=data1,ax=axis1)
    axis1.set_title('pclass vs fare survived comparision')

    sns.violinplot(x='Pclass',y='Age',hue='Survived',data=data1,split=True,ax=axis2)
    axis2.set_title('pclass vs age survival comparison')

    sns.boxplot(x='Pclass',y='FamilySize',hue='Survived',data=data1,ax=axis3)
    axis3.set_title('pclass vs family size survived comparsion')

    if is_show:
        plt.show()

    # 比较Sex和其他变量
    fig,qaxis = plt.subplots(1,3,figsize=(14,12))

    sns.barplot(x='Sex',y='Survived',hue='Embarked',data=data1,ax=qaxis[0])
    qaxis[0].set_title('sex vs embarked survived comparsion')

    sns.barplot(x='Sex',y='Survived',hue='Pclass',data=data1,ax=qaxis[1])
    qaxis[1].set_title('sex vs pclass sirvival comparsion')

    sns.barplot(x='Sex',y='Survived',hue='IsAlone',data=data1,ax=qaxis[2])
    qaxis[2].set_title('sex vs isalone survival comparsion')

    if is_show:
        plt.show()

    # 更多的比较
    fig,(maxis1,maxis2) = plt.subplots(1,2,figsize=(14,12))

    # familysize、sex、survival比较
    sns.pointplot(x='FamilySize',y='Survived',hue='Sex',data=data1,
                  palette={'male':'blue','female':'pink'},markers=['*','o'],
                  linestyles=['-','--'],ax=maxis1)

    # pclass、sex、survival比较

    sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data1,
                  palette={"male": "blue", "female": "pink"},
                  markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)
    if is_show:
        plt.show()

    # embarked pclass sex survival 比较
    e = sns.FacetGrid(data1,col='Embarked')
    e.map(sns.pointplot,'Pclass','Survived','Sex',ci=95.0,palette='deep')
    e.add_legend()
    if is_show:
        plt.show()

    # 绘制年龄和生存率的分布图
    a = sns.FacetGrid(data1,hue='Survived',aspect=4)
    a.map(sns.kdeplot,'Age',shade=True)
    a.set(xlim=(0,data1['Age'].max()))
    a.add_legend()
    if is_show:
        plt.show()

    # sex pclass age 直方图比较
    h = sns.FacetGrid(data1,row='Sex',col='Pclass',hue='Survived')
    h.map(plt.hist,'Age',alpha=0.75)
    h.add_legend()
    if is_show:
        plt.show()

    # 整个数据集的散点图矩阵
    pp = sns.pairplot(data1, size=1,
                      diag_kind='kde',diag_kws=dict(shade=True),plot_kws=dict(s=10))
    pp.set(xticklabels=[])
    if is_show:
        plt.show()
    correlation_heatmap(data1)

# 绘制关联关系热度图
def correlation_heatmap(df,is_show=True):
    _,ax = plt.subplots(figsize=(14,12))
    colormap = sns.diverging_palette(220,10,as_cmap=True)
    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink':0.9},
        ax=ax,
        annot=True,
        linewidths=0.1,vmax=1.0,linecolor='white',
        annot_kws={'fontsize':12}
    )
    plt.title('pearson correlation of features',y=1.05,size=15)
    if is_show:
        plt.show()

# 使用不同的算法训练模型
def train_models():
    # 模型数据
    MLA = [
        # 集合方法
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),

        # 高斯处理
        gaussian_process.GaussianProcessClassifier(),

        # GLM
        linear_model.LogisticRegressionCV(),
        linear_model.PassiveAggressiveClassifier(),
        linear_model.RidgeClassifierCV(),
        linear_model.SGDClassifier(),
        linear_model.Perceptron(),

        # 朴素贝叶斯
        neighbors.KNeighborsClassifier(),

        # svm
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(),

        # tree
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),

        # 判别分析
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),

        # xgboost
        XGBClassifier()
    ]

    # 划分训练集对象
    cv_split = model_selection.ShuffleSplit(n_splits=10,test_size=0.3,train_size=0.6,random_state=0)
    # 创建一张表格比较 MLA 的度量
    MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
    MLA_compare = pd.DataFrame(columns=MLA_columns)

    # 创建表，比较MLA的预测结果和真实值
    MLA_predict = data1[Target]

    # 遍历 MLA 并保存性能到表格
    row_index = 0
    for alg in MLA:
        # 设置名字和参数
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index,'MLA Name'] = MLA_name
        MLA_compare.loc[row_index,'MLA Parameters'] = str(alg.get_params())

        # 使用交叉验证评估模型
        cv_results = model_selection.cross_validate(alg,data1[data1_x_bin],data1[Target],cv=cv_split)

        MLA_compare.loc[row_index,'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()

        # 测试评分三倍标准差
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std() * 3

        # 保存MLA 预测
        alg.fit(data1[data1_x_bin],data1[Target])
        MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])

        row_index += 1

    # 打印排序好的表格
    MLA_compare.sort_values(by=['MLA Test Accuracy Mean'],ascending=False,inplace=True)

    # 使用柱状图展示表格
    sns.barplot(x='MLA Test Accuracy Mean',y='MLA Name',data=MLA_compare,color='m')
    plt.title('machine learning algorithm accuracy score\n')
    plt.xlabel('accuracy score(%)')
    plt.ylabel('algorithm')
    plt.show()
    correlation_heatmap(MLA_predict)

# 基于一个模型进行调参
def tune_model():
    # 划分训练集对象
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)

    # 基础模型
    dtree = tree.DecisionTreeClassifier(random_state=0)
    base_results = model_selection.cross_validate(dtree,data1[data1_x_bin],data1[Target])
    dtree.fit(data1[data1_x_bin],data1[Target])
    print('BEFORE DT Parameters: ', dtree.get_params())
    print("BEFORE DT Training w/bin score mean: {:.2f}".format(base_results['train_score'].mean() * 100))
    print("BEFORE DT Test w/bin score mean: {:.2f}".format(base_results['test_score'].mean() * 100))
    print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}".format(base_results['test_score'].std() * 100 * 3))
    print('-' * 10)

    # 通过网格搜索调参
    grid_param = {
        'criterion':['gini','entropy'],
        'max_depth':[2,4,6,8,10,None],
        'random_state':[0]
    }
    tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(),
                                              param_grid=grid_param,
                                              scoring='roc_auc',cv=cv_split)

    tune_model.fit(data1[data1_x_bin],data1[Target])
    print('AFTER DT Parameters: ', tune_model.best_params_)
    print("AFTER DT Training w/bin score mean: {:.2f}".format(
        tune_model.cv_results_['mean_train_score'][tune_model.best_index_] * 100))
    print("AFTER DT Test w/bin score mean: {:.2f}".format(
        tune_model.cv_results_['mean_test_score'][tune_model.best_index_] * 100))
    print("AFTER DT Test w/bin score 3*std: +/- {:.2f}".format(
        tune_model.cv_results_['std_test_score'][tune_model.best_index_] * 100 * 3))
    print('-' * 10)

    # 训练模型-基于特征选择
    print('BEFORE DT RFE Training Shape Old: ', data1[data1_x_bin].shape)
    print('BEFORE DT RFE Training Columns Old: ', data1[data1_x_bin].columns.values)

    print("BEFORE DT RFE Training w/bin score mean: {:.2f}".format(base_results['train_score'].mean() * 100))
    print("BEFORE DT RFE Test w/bin score mean: {:.2f}".format(base_results['test_score'].mean() * 100))
    print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}".format(base_results['test_score'].std() * 100 * 3))
    print('-' * 10)
    ## 特征选择
    dtree_rfe = feature_selection.RFECV(dtree,step=1,scoring='accuracy',cv=cv_split)
    dtree_rfe.fit(data1[data1_x_bin],data1[Target])
    # 训练模型
    X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()]
    rfe_results = model_selection.cross_validate(dtree,data1[X_rfe],data1[Target],
                                                 cv=cv_split)
    print('AFTER DT RFE Training Shape New: ', data1[X_rfe].shape)
    print('AFTER DT RFE Training Columns New: ', X_rfe)

    print("AFTER DT RFE Training w/bin score mean: {:.2f}".format(rfe_results['train_score'].mean() * 100))
    print("AFTER DT RFE Test w/bin score mean: {:.2f}".format(rfe_results['test_score'].mean() * 100))
    print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}".format(rfe_results['test_score'].std() * 100 * 3))
    print('-' * 10)

    rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=grid_param,
                                                  scoring='roc_auc', cv=cv_split)
    rfe_tune_model.fit(data1[X_rfe], data1[Target])
    print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)
    print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}".format(
        rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_] * 100))
    print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}".format(
        rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_] * 100))
    print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}".format(
        rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_] * 100 * 3))
    print('-' * 10)

    # 图形化树
    dot_data = tree.export_graphviz(dtree,out_file=None,feature_names=data1_x_bin,
                                    class_names=True,filled=True,rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("tree")

# 集成多个算法
def validate_implement():
    # 查看多个算法的关联关系热度图
    # train_models()
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)
    # 根据热度图把关联关系为1的模型去掉，把剩余的构造为一个投票器
    vote_est = [
        # 集合方法
        ('ada',ensemble.AdaBoostClassifier()),
        ('bc',ensemble.BaggingClassifier()),
        ('etc',ensemble.ExtraTreesClassifier()),
        ('gbc',ensemble.GradientBoostingClassifier()),
        ('rfc',ensemble.RandomForestClassifier()),

        # 高斯方法
        ('gpc',gaussian_process.GaussianProcessClassifier()),

        # GLM
        ('lr',linear_model.LogisticRegressionCV()),

        # 朴素贝叶斯
        ('bnb',naive_bayes.BernoulliNB()),
        ('gnb',naive_bayes.GaussianNB()),

        # 最近邻
        ('knn',neighbors.KNeighborsClassifier()),

        # svm
        ('svc',svm.SVC(probability=True)),

        # xgboost
        ('xgb',XGBClassifier())
    ]
    # 硬投票/多数决定
    vote_hard = ensemble.VotingClassifier(estimators=vote_est,voting='hard')
    vote_hard_cv = model_selection.cross_validate(vote_hard,data1[data1_x_bin],data1[Target],
                                                  cv=cv_split)
    vote_hard.fit(data1[data1_x_bin],data1[Target])
    print("Hard Voting Training w/bin score mean: {:.2f}".format(vote_hard_cv['train_score'].mean() * 100))
    print("Hard Voting Test w/bin score mean: {:.2f}".format(vote_hard_cv['test_score'].mean() * 100))
    print("Hard Voting Test w/bin score 3*std: +/- {:.2f}".format(vote_hard_cv['test_score'].std() * 100 * 3))
    print('-' * 10)

    # 软投票/权重概率
    vote_soft = ensemble.VotingClassifier(estimators=vote_est,voting='soft')
    vote_soft_cv = model_selection.cross_validate(vote_soft,data1[data1_x_bin],data1[Target],
                                                  cv=cv_split)
    vote_soft.fit(data1[data1_x_bin],data1[Target])
    print("Soft Voting Training w/bin score mean: {:.2f}".format(vote_soft_cv['train_score'].mean() * 100))
    print("Soft Voting Test w/bin score mean: {:.2f}".format(vote_soft_cv['test_score'].mean() * 100))
    print("Soft Voting Test w/bin score 3*std: +/- {:.2f}".format(vote_soft_cv['test_score'].std() * 100 * 3))
    print('-' * 10)

    # 使用网格搜索调参
    grid_n_estimator = [10, 50, 100, 300]
    grid_ratio = [.1, .25, .5, .75, 1.0]
    grid_learn = [.01, .03, .05, .1, .25]
    grid_max_depth = [2, 4, 6, 8, 10, None]
    grid_min_samples = [5, 10, .03, .05, .10]
    grid_criterion = ['gini', 'entropy']
    grid_bool = [True, False]
    grid_seed = [0]

    grid_param = [
        [{
            # AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
            'n_estimators': grid_n_estimator,  # default=50
            'learning_rate': grid_learn,  # default=1
            # 'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
            'random_state': grid_seed
        }],

        [{
            # BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
            'n_estimators': grid_n_estimator,  # default=10
            'max_samples': grid_ratio,  # default=1.0
            'random_state': grid_seed
        }],

        [{
            # ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
            'n_estimators': grid_n_estimator,  # default=10
            'criterion': grid_criterion,  # default=”gini”
            'max_depth': grid_max_depth,  # default=None
            'random_state': grid_seed
        }],

        [{
            # GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
            # 'loss': ['deviance', 'exponential'], #default=’deviance’
            'learning_rate': [.05],
        # default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            'n_estimators': [300],
        # default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            # 'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
            'max_depth': grid_max_depth,  # default=3
            'random_state': grid_seed
        }],

        [{
            # RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
            'n_estimators': grid_n_estimator,  # default=10
            'criterion': grid_criterion,  # default=”gini”
            'max_depth': grid_max_depth,  # default=None
            'oob_score': [True],
        # default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
            'random_state': grid_seed
        }],

        [{
            # GaussianProcessClassifier
            'max_iter_predict': grid_n_estimator,  # default: 100
            'random_state': grid_seed
        }],

        [{
            # LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
            'fit_intercept': grid_bool,  # default: True
            # 'penalty': ['l1','l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # default: lbfgs
            'random_state': grid_seed
        }],

        [{
            # BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
            'alpha': grid_ratio,  # default: 1.0
        }],

        # GaussianNB -
        [{}],

        [{
            # KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
            'n_neighbors': [1, 2, 3, 4, 5, 6, 7],  # default: 5
            'weights': ['uniform', 'distance'],  # default = ‘uniform’
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }],

        [{
            # SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
            # http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
            # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [1, 2, 3, 4, 5],  # default=1.0
            'gamma': grid_ratio,  # edfault: auto
            'decision_function_shape': ['ovo', 'ovr'],  # default:ovr
            'probability': [True],
            'random_state': grid_seed
        }],

        [{
            # XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
            'learning_rate': grid_learn,  # default: .3
            'max_depth': [1, 2, 4, 6, 8, 10],  # default 2
            'n_estimators': grid_n_estimator,
            'seed': grid_seed
        }]
    ]

    start_total = time.perf_counter()
    for clf,param in zip(vote_est,grid_param):
        start = time.perf_counter()
        best_search = model_selection.GridSearchCV(estimator=clf[1],param_grid=param,
                                                   cv=cv_split,scoring='roc_auc')
        best_search.fit(data1[data1_x_bin],data1[Target])
        run = time.perf_counter()-start

        best_param = best_search.best_params_
        print('the best parameter for {} is {} with a runtime of{:.2f} seconds'.format(
            clf[1].__class__.__name__,best_param,run))
        clf[1].set_params(**best_param)

    run_total = time.perf_counter() - start_total
    print('Total optimization time was {:.2f} minutes.'.format(run_total / 60))

    print('-' * 10)

    # submit
    data_val['Survived'] = vote_hard.predict(data_val[data1_x_bin])
    submit = data_val[['PassengerId','Survived']]
    submit.to_csv("./submit.csv", index=False)


validate_implement()