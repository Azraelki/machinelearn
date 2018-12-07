import numpy as np
import sys

class NeuralNetMLP:
    '''
    前馈神经网络/多层感知器 分类器
    '''
    def __init__(self,n_hidden=30,l2=0.,epochs=100,eta=0.001,shuffle=True,
                 minibatch_size=1,seed=None,check=False):
        '''
        :param n_hidden: 隐藏单元的个数
        :param l2: 正则参数
        :param epochs: 迭代次数
        :param eta: 学习率
        :param shuffle: 每次迭代是否重置数据集
        :param minibatch_size: 每次mini批处理的样本数量
        :param seed: 随机初始化参数和重置样本集顺序时的种子数，方便重现结果
        :param check: 是否开启梯度检测
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
        self.check = check

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

    def _forward(self,X,w_h,w_out):
        '''
        计算前向传播
        :param X:
        :return:
        '''
        # step 1: 隐藏层的净输入值
        # [n_samples,n_features] dor [n_features,n_hidden]
        # -> [n_samples,n_hidden]
        z_h = np.dot(X,w_h) + self.b_h

        # step 2: 隐层激活
        a_h = self._sigmoid(z_h)

        # step 3: 输出层净输入值
        # [n_samples,n_hidden] dot [n_hidden,n_classlabels]
        # -> [n_samples,n_classlabels]
        z_out = np.dot(a_h,w_out) + self.b_out

        # step 4: 激活输出层
        a_out = self._sigmoid(z_out)

        return z_h,a_h,z_out,a_out

    def _compute_cost(self,y_enc,output,w_h,w_out):
        '''
        计算成本函数
        :param y_enc: one-hot形式的类标签
        :param output: 前向传播返回的a_out
        :return: 正则后的成本值
        '''
        L2_TERM = (self.l2*(np.sum(w_h**2.)+np.sum(w_out**2.))) # 正则项
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
        z_h,a_h,z_out,a_out = self._forward(X,self.w_h,self.w_out)
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
                z_h,a_h,z_out,a_out = self._forward(X_train[batch_idx],self.w_h,self.w_out)

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

                if self.check:
                    # 检验梯度-begin
                    grad_diff = self._gradient_checking(X_train[batch_idx],
                                                        y_train_enc[batch_idx],
                                                        self.w_h,self.w_out,
                                                        1e-5,grad_w_h,grad_w_out)
                    if grad_diff <= 1e-7:
                        print("OK: %s"%grad_diff)
                    elif grad_diff <= 1e-4:
                        print("Warning: %s"%grad_diff)
                    else:
                        print("Problem: %s"%grad_diff)
                    # 检验梯度-end


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
            z_h,a_h,z_out,a_out = self._forward(X_train,self.w_h,self.w_out)

            cost = self._compute_cost(y_enc=y_train_enc,output=a_out,w_h=self.w_h,w_out=self.w_out)

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

    def _gradient_checking(self,X,y_enc,w_h,w_out,epsilon,grad_h,grad_out):
        '''
        梯度检验（计算成本较大，只在debug少量数据时检验梯度计算是否正确）
        :param X:
        :param y_enc:
        :param w_h:
        :param w_out:
        :param epsilon:
        :param grad_h:
        :param grad_out:
        :return: relative_error: 数值梯度（数值逼近的梯度）和解析梯度（反向传播计算的梯度）的相对误差
        '''
        # 计算w_h的数值梯度
        num_grad1 = np.zeros(np.shape(w_h))
        epsilon_ary1 = np.zeros(np.shape(w_h))

        for i in range(w_h.shape[0]): # 计算每个权重的数值偏导
            for j in range(w_h.shape[1]):
                epsilon_ary1[i,j] = epsilon
                z_h,a_h,z_out,a_out = self._forward(X,w_h-epsilon_ary1,w_out)
                cost1 = self._compute_cost(y_enc,a_out,w_h-epsilon_ary1,w_out)

                z_h, a_h, z_out, a_out = self._forward(X, w_h + epsilon_ary1, w_out)
                cost2 = self._compute_cost(y_enc, a_out, w_h + epsilon_ary1, w_out)

                num_grad1[i,j] = (cost2-cost1)/(2*epsilon)
                epsilon_ary1[i,j] = 0

        # 计算w_out的数值梯度
        num_grad2 = np.zeros(np.shape(w_out))
        epsilon_ary2 = np.zeros(np.shape(w_out))

        for i in range(w_out.shape[0]):  # 计算每个权重的数值偏导
            for j in range(w_out.shape[1]):
                epsilon_ary2[i, j] = epsilon
                z_h, a_h, z_out, a_out = self._forward(X, w_h, w_out - epsilon_ary2)
                cost1 = self._compute_cost(y_enc, a_out, w_h, w_out - epsilon_ary2)

                z_h, a_h, z_out, a_out = self._forward(X, w_h, w_out + epsilon_ary2)
                cost2 = self._compute_cost(y_enc, a_out, w_h, w_out + epsilon_ary2)

                num_grad2[i, j] = (cost2 - cost1) / (2 * epsilon)
                epsilon_ary2[i, j] = 0

        num_grad = np.hstack((num_grad1.flatten(),num_grad2.flatten())) #数值梯度 化成一维，方便计算

        grad = np.hstack((grad_h.flatten(),grad_out.flatten())) # 解析梯度

        # 归一化处理，防止因尺度变化造成的波动
        norm1 = np.linalg.norm(num_grad-grad)
        norm2 = np.linalg.norm(num_grad)
        norm3 = np.linalg.norm(grad)
        relative_error = norm1/(norm2+norm3)
        return relative_error
















