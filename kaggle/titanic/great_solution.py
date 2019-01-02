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
# from pandas.tools.plotting import scatter_matrix

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
def correlation_heatmap(df):
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
a = MLA_compare
pass
