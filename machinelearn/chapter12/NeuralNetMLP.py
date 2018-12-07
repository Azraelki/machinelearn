import numpy as np
import sys

class NeuralNetMLP:
    '''
    前馈神经网络/多层感知器 分类器
    '''
    def __init__(self,n_hidden=30,l2=0.,epochs=100,eta=0.001,shuffle=True,
                 minibatch_size=1,seed=None):
        '''
        :param n_hidden: 隐藏单元的个数
        :param l2: 正则参数
        :param epochs: 迭代次数
        :param eta: 学习率
        :param shuffle: 每次迭代是否重置数据集
        :param minibatch_size: 每次mini批处理的样本数量
        :param seed: 随机初始化参数和重置样本集顺序时的种子数，方便重现结果
        attributes
        -----------
        eval_ : dict
            收集成本值，训练精准度，每次迭代的验证精准度
        '''
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self,y,n_classes):
        '''
        把类别表示成one-hot形式
        :param y:
        :param n_classes:
        :return:
        '''
        onehot = np.zeros((n_classes,y.shape[0]))
        for idx,val in enumerate(y.astype(int)):
            onehot[val,idx] = 1
        return onehot.T

    def _sigmoid(self,z):
        '''
        激励函数
        :param z: 净输入值
        :return:
        '''
        return 1.0/(1.0+np.exp(-np.clip(z,-250,250)))

    def _forward(self,X):
        '''
        计算前向传播
        :param X:
        :return:
        '''
        # step 1: 隐藏层的净输入值
        # [n_samples,n_features] dor [n_features,n_hidden]
        # -> [n_samples,n_hidden]
        z_h = np.dot(X,self.w_h) + self.b_h

        # step 2: 隐层激活
        a_h = self._sigmoid(z_h)

        # step 3: 输出层净输入值
        # [n_samples,n_hidden] dot [n_hidden,n_classlabels]
        # -> [n_samples,n_classlabels]
        z_out = np.dot(a_h,self.w_out) + self.b_out

        # step 4: 激活输出层
        a_out = self._sigmoid(z_out)

        return z_h,a_h,z_out,a_out

    def _compute_cost(self,y_enc,output):
        '''
        计算成本函数
        :param y_enc: one-hot形式的类标签
        :param output: 前向传播返回的a_out
        :return: 正则后的成本值
        '''
        L2_TERM = (self.l2*(np.sum(self.w_h**2.)+np.sum(self.w_out**2.))) # 正则项
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1-output)
        cost = np.sum(term1-term2) + L2_TERM
        return cost

    def predict(self,X):
        '''
        预测函数
        :param X:
        :return:
        '''
        z_h,a_h,z_out,a_out = self._forward(X)
        y_pred = np.argmax(z_out,axis=1)
        return y_pred

    def fit(self,X_train,y_train,X_valid,y_valid):
        '''
        从训练集中训练模型
        :param X_train:
        :param y_train:
        :param X_valid:
        :param y_valid:
        :return:
        '''
        n_output = np.unique(y_train).shape[0] # 类别数量

        n_features = X_train.shape[1] # 特征数量

        ## 权重初始化
        # input -> hidden
        self.b_h = np.zeros(self.n_hidden) # 输入层偏置单元
        self.w_h = self.random.normal(loc=0.0,scale=0.1,size=(n_features,self.n_hidden))

        # hidden -> output
        self.b_out = np.zeros(n_output) # 隐藏层偏置单元
        self.w_out = self.random.normal(loc=0.0,scale=0.1,size=(self.n_hidden,n_output))

        epoch_strlen = len(str(self.epochs)) # 迭代次数的字符串长度
        self.eval_ = {'cost':[],'train_acc':[],'valid_acc':[]}

        y_train_enc = self._onehot(y_train,n_output) # 标签转化为one-hot

        # 迭代训练
        for i in range(self.epochs):
            # minibach迭代
            indices = np.arange(X_train.shape[0])
            if self.shuffle: # 每次迭代训练时是否重置顺序
                self.random.shuffle(indices)
            for start_idx in range(0,indices.shape[0]-self.minibatch_size+1,self.minibatch_size):
                batch_idx  = indices[start_idx:start_idx+self.minibatch_size] # 获取minibatch下标

                # 前向传播
                z_h,a_h,z_out,a_out = self._forward(X_train[batch_idx])

                # 后向传播
                # [n_samples,n_classlabels]
                sigma_out = a_out - y_train_enc[batch_idx] # 计算输出层的误差向量

                # [n_samples,n_hidden] sigmoid_derivative_h为sigmoid函数的导数项
                sigmoid_derivative_h = a_h*(1.-a_h) # 计算隐层的误差项 ----1

                # [n_samples,n_classlabels] dot [n_classlabels,n_hidden]
                # -> [n_samples,n_hidden] 隐层误差=（输出层误差 dot 隐层权重系数）* 激活函数的导数项
                sigma_h = (np.dot(sigma_out,self.w_out.T) * sigmoid_derivative_h) # 计算隐层的误差项 ----2

                # [n_features,n_samples] dot [n_samples,n_hidden]
                # -> [n_features,n_hidden] 计算：隐层偏导 = 输入层 dot 隐层误差
                grad_w_h = np.dot(X_train[batch_idx].T,sigma_h)
                grad_b_h = np.sum(sigma_h,axis=0)

                # [n_hidden,n_samples] dot [n_samples,n_classlabels]
                # -> [n_hidden,n_classlabels] 计算：输出层偏导 = 隐层激励值 dot 输出层误差
                grad_w_out = np.dot(a_h.T,sigma_out)
                grad_b_out = np.sum(sigma_out,axis=0)

                # 正则化 并且更新隐藏层和输出层权重
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h # 偏置单元无需正则
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out +self.l2*self.w_out)
                delta_b_out = grad_b_out # 偏置单元无需正则
                self.w_out -= self.eta*delta_w_out
                self.b_out -= self.eta*delta_b_out

            ## 评估
            # 每次迭代之后进行评估
            z_h,a_h,z_out,a_out = self._forward(X_train)

            cost = self._compute_cost(y_enc=y_train_enc,output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train==y_train_pred)).astype(np.float)/X_train.shape[0])
            valid_acc = ((np.sum(y_valid==y_valid_pred)).astype(np.float)/X_valid.shape[0])

            sys.stderr.write("\r%0*d/%d | cost: %.2f | train/valid acc: %.2f%%/%.2f%%"%
                             (epoch_strlen,i+1,self.epochs,cost,train_acc*100,valid_acc*100))

            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self




