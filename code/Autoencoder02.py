# coding = utf-8

import tensorflow as tf
import pickle
import math
import numpy as np
from sklearn.model_selection import KFold
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK
from sklearn import model_selection
import hyperopt
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# from tensorflow.python.training.momentum import MomentumOptimizer


class AutoEncoder(tf.keras.Model):

    def __init__(self, data1, data2, testdata1=None, testdata2=None, n_hiddensh=1, activation=None):
        super(AutoEncoder, self).__init__()
        # training datasets
        self.l5 = None
        self.l4 = None
        self.l3 = None
        self.l6 = None
        self.l2 = None
        self.l1 = None
        self.training_data1 = data1
        self.training_data2 = data2
        # test datasets
        self.test_data1 = testdata1
        self.test_data2 = testdata2
        # number of features
        self.n_input1 = data1.shape[1]
        self.n_input2 = data2.shape[1]
        self.n_hiddensh = n_hiddensh
        # activation function
        #print("shape of n_hiddensh", n_hiddensh)
        self.activation = activation

    def encode(self, X1, X2):
        # =============================================================================
        # first hidden layer composed of two parts related to two sources (X1, X2)
        # - build a fully connected layer
        # - apply the batch normalization
        # - apply an activation function
        # =============================================================================
        self.l1_layer=tf.keras.layers.Dense(self.n_hidden1, kernel_initializer=self._init, name='layer1')
        l1 = self.l1_layer(X1)

        l1 = tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))
        l1 = self.activation(l1)
        self.n_layer1 = l1.shape
        self.l1=l1
       # print("layer1 shape", l1.shape)
        self.l2_layer = tf.keras.layers.Dense(self.n_hidden2, kernel_initializer=self._init, name='layer2')
        l2 = self.l2_layer(X2)
        l2 = tf.keras.layers.BatchNormalization()(l2, training=bool(self.is_train))
        l2 = self.activation(l2)
        self.n_layer2 = l2.shape
        self.l2=l2
        #print("layer2 shape", l2.shape)
        # =============================================================================
        # fuse the parts of the first hidden layer
        # =============================================================================
        self.l3_layer= tf.keras.layers.Dense(self.n_hiddensh, kernel_initializer=self._init, name='layer3')
        l3 = self.l3_layer(tf.concat([l1, l2], 1))
        #l1,l2 are bounded in l
        #print("-----layer3 shape", l3.shape)
        l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))
        self.n_layer3 = l3.shape
        self.l3=l3
        #print("layer3 shape", l3.shape)
        return l3

    def decode(self, H):
        self.l4_layer=tf.keras.layers.Dense(self.n_hidden1 + self.n_hidden2, kernel_initializer=self._init, name='layer4')
        l4 = self.l4_layer(H)
        l4 = tf.keras.layers.BatchNormalization()(l4, training=bool(self.is_train))
        self.n_layer4 = l4.shape
        self.l4=l4
        #print("layer4 shape", l4.shape)
        s1, s2 = tf.split(l4, [self.n_hidden1, self.n_hidden2], 1)
        self.l5_layer=tf.keras.layers.Dense(self.n_input1, kernel_initializer=self._init, name='layer5')
        l5 = self.l5_layer(s1)
        l5 = tf.keras.layers.BatchNormalization()(l5, training=bool(self.is_train))
        l5 = self.activation(l5)#l1
        self.n_layer5 = l5.shape
        self.l5=l5
        #print("layer5 shape", l5.shape)
        self.l6_layer=tf.keras.layers.Dense(self.n_input2, kernel_initializer=self._init, name='layer6')
        l6 = self.l6_layer(s2)
        l6 = tf.keras.layers.BatchNormalization()(l6, training=bool(self.is_train))
        l6 = self.activation(l6)#l2
        self.n_layer6 = l6.shape
        self.l6=l6
        #print("layer6 shape", l6.shape)
        return l5, l6


    #def get_weights(self):
       # self.W1 = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=self.n_layer1), trainable=True,
       #                       name="layer1/kernel")
       # self.W2 = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=self.n_layer2), trainable=True,
       #                       name="layer2/kernel")
       #self.Wsh = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=self.n_layer3), trainable=True,
       #                        name="layer3/kernel")
       # self.Wsht = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=self.n_layer4), trainable=True,
       #                        name="layer4/kernel")
       #  self.W1t = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=self.n_layer5), trainable=True,
       #                         name="layer5/kernel")
       #   self.W2t = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=self.n_layer6), trainable=True,
       #    #                      name="layer6/kernel")

    def get_weights(self):

        self.W1 = self.l1_layer.kernel
        self.W2 = self.l2_layer.kernel
        self.Wsht = self.l3_layer.kernel
        self.Wsh = self.l4_layer.kernel
        self.W1t = self.l5_layer.kernel
        self.W2t = self.l6_layer.kernel

    def L1regularization(self, weights):
        return tf.reduce_sum(tf.abs(weights))

    def L2regularization(self, weights, nbunits):
        return math.sqrt(nbunits) * tf.nn.l2_loss(weights)

    def loss(self, X1, X2):

        self.H = self.encode(X1, X2)
        X1_, X2_ = self.decode(self.H)
        self.get_weights()

        # sparse group lasso
        sgroup_lasso = self.L2regularization(self.W1, self.n_input1 * self.n_hidden1) + \
                       self.L2regularization(self.W2, self.n_input2 * self.n_hidden2)

        # lasso
        lasso = self.L1regularization(self.W1) + self.L1regularization(self.W2) + \
                self.L1regularization(self.Wsh) + self.L1regularization(self.Wsht) + \
                self.L1regularization(self.W1t) + self.L1regularization(self.W2t)

        # reconstruction Error
        error = tf.losses.mean_squared_error(X1, X1_) + tf.losses.mean_squared_error(X2, X2_)

        # Loss function
        cost = 0.5 * error + 0.5 * self.lamda * (1 - self.alpha) * sgroup_lasso + 0.5 * self.lamda * self.alpha * lasso
        return cost

    def train(self, train_index, val_index):
        # training data
        train_input1 = self.training_data1[train_index, :]#370 x 4/5
        train_input2 = self.training_data2[train_index, :]
        #print("The size of  train_input1",self.training_data1.shape)
        print("---------------TRain starts---------")
        # validation data
        val_input1 = self.training_data1[val_index, :]#370 x 1/5
        val_input2 = self.training_data2[val_index, :]

        #save_sess = self.sess

        # costs history:
        costs = []
        costs_val = []
        costs_val_inter = []
        costs_inter = []

        # for early stopping:
        best_cost = 0
        best_val_cost = 100000
        stop = False
        last_improvement = 0

        n_samples = train_input1.shape[0]  # size of the training set #370 x 4/5
        vn_samples = val_input1.shape[0]  # size of the validation set#370 x 1/5

        # train the mini_batches model using the early stopping criteria
        #for var in self.trainable_variables:
           #print("The name of parameter and shape", var.name, var.shape)

        epoch = 0
        while epoch < self.max_epochs and stop == False:
            # train the model on the training set by mini batches
            # shuffle then split the training set to mini-batches of size self.batch_size
            seq = list(range(n_samples))#370 x 4/5
            #print("The number of n_samples",n_samples)#370 x 4/5
            random.shuffle(seq)
            mini_batches = [
                seq[k:k + self.batch_size]
                for k in range(0, n_samples, self.batch_size)
            ]

            avg_cost = 0.  # the average cost of mini_batches

            for sample in mini_batches:
                batch_xs1 = train_input1[sample][:]
                batch_xs2 = train_input2[sample][:]

                #feed_dictio = {self.X1: batch_xs1, self.X2: batch_xs2, self.is_train: True}
                #cost = self.sess.run([self.loss_, self.train_step], feed_dict=feed_dictio)
                #avg_cost += cost[0] * len(sample) / n_samples

                with tf.GradientTape() as tape:
                    # 计算当前批次的损失
                    current_loss = self.loss(batch_xs1, batch_xs2)

                # 计算梯度
                gradients = tape.gradient(current_loss, self.trainable_variables)
                #print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
                #print(gradients)
                #print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
                # 更新模型的参数
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                cost = self.loss(batch_xs1, batch_xs2)
                #print("-----------------------Kosten---------------------------",cost)
                #print("---------------------------------------------")
                avg_cost += cost[0] * len(sample) / n_samples


            # train the model on the validation set by mini batches
            # split the validation set to mini-batches of size self.batch_size

            #-------------------------------------------------------------------#

            seq = list(range(vn_samples))
            mini_batches = [
                seq[k:k + self.batch_size]
                for k in range(0, vn_samples, self.batch_size)
            ]
            avg_cost_val = 0.

            for sample in mini_batches:
                batch_xs1 = val_input1[sample][:]
                batch_xs2 = val_input2[sample][:]

                #feed_dictio = {self.X1: batch_xs1, self.X2: batch_xs2, self.is_train: False}
                cost_val = self.loss(batch_xs1, batch_xs2)
                avg_cost_val += cost_val[0] * len(sample) / vn_samples

            # cost history since the last best cost
            costs_val_inter.append(avg_cost_val)
            costs_inter.append(avg_cost)

            # early stopping based on the validation set/ max_steps_without_decrease of the loss value: require_improvement
            if avg_cost_val < best_val_cost:
                #save_sess = self.sess  # save session
                best_val_cost = avg_cost_val
                best_cost = avg_cost
                costs_val += costs_val_inter  # costs history of the validation set
                costs += costs_inter  # costs history of the training set
                last_improvement = 0
                costs_val_inter = []
                costs_inter = []
            else:
                last_improvement += 1
            if last_improvement > self.require_improvement:
                # print("No improvement found during the (self.require_improvement) last iterations, stopping optimization.")
                # Break out from the loop.
                stop = True
                #self.sess = save_sess  # restore session with the best cost

            epoch += 1

        self.histcosts = costs
        self.histvalcosts = costs_val
        return best_cost, best_val_cost

    # def train_step(self, X1, X2):
    # with tf.GradientTape() as tape:
    #    # 假设 self.loss_ 是一个方法，计算给定输入的损失
    #     loss_value = self.loss(X1, X2)

    # # 计算损失关于模型可训练参数的梯度
    #  gradients = tape.gradient(loss_value, self.model.trainable_variables)

    #  # 使用优化器应用梯度（更新模型参数）
    # self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    # return loss_value

    def cross_validation(self, params):
        # retrieve parameters
        self.batch_size = params['batch_size']
        #print("self.batch_size",self.batch_size)#16
        self.n_hidden1 = params['units1']
        self.n_hidden2 = params['units2']
        self.alpha = params['alpha']
        self.lamda = params['lamda']
        self.learning_rate = params['learning_rate']

        # k fold validation
        k = 5
        self.require_improvement = 20
        self.max_epochs = 1000
        init = params['initializer']
        if init == 'normal':
            self._init = tf.random_normal_initializer
        if init == 'uniform':
            self._init = tf.random_uniform_initializer
        if init == 'He':
            self._init = tf.keras.initializers.HeNormal()
        if init == 'xavier':
            self._init = tf.keras.initializers.GlorotNormal()

        opt = params['optimizer']
       # if opt == 'SGD':
           # self.optimizer = tf.keras.optimizers.SGD()
        #if opt == 'adam':
        #    self.optimizer = tf.keras.optimizers.Adam()
        #if opt == 'nadam':
        #    self.optimizer = tf.keras.optimizers.Nadam()
       # if opt == 'Momentum':
       #     self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate,momentum=0.9)
        #if opt == 'RMSProp':
        #    self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        if opt == 'SGD':
                # self.optimizer = tf.keras.optimizers.SGD()
            self.optimizer = tf.keras.optimizers.legacy.SGD()
        if opt == 'adam':
                # self.optimizer = tf.keras.optimizers.Adam()
            self.optimizer = tf.keras.optimizers.legacy.Adam()
        if opt == 'nadam':
                # self.optimizer = tf.keras.optimizers.Nadam()
            self.optimizer = tf.keras.optimizers.legacy.Nadam()
        if opt == 'Momentum':
                # self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate,momentum=0.9)
        if opt == 'RMSProp':
                # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
            self.optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=self.learning_rate)

        # cross-validation
        data = np.concatenate([self.training_data1, self.training_data2], axis=1)
        #print("data shape is ",data.shape)
        kf = KFold(n_splits=k, shuffle=True)  # k fold cross validation
        kf.get_n_splits(data)  # returns the number of splitting iterations in the cross-validator
        #data is component with 301+301
        # validation set loss
        loss_cv = 0
        val_loss_cv = 0
        #counter=0
        #print("train check self.trainable_variables()",self.trainable_variables())





        for train_index, val_index in kf.split(data):
            #counter+=1
            print("train_index",train_index.shape)
            print("val_index",val_index.shape)
            # reset tensor graph after each cross_validation run
            # tf.reset_default_graph()
            #model = AutoEncoder(self.training_data1, self.training_data2, self.test_data1, self.test_data2,
                          #      self.n_hiddensh,
                          #      self.activation)
            # self.X1 = tf.placeholder("float", shape=[None, self.training_data1.shape[1]])
            # self.X2 = tf.placeholder("float", shape=[None, self.training_data2.shape[1]])

            #self.X1 = tf.Variable(np.zeros(shape=(self.batch_size, self.training_data1.shape[1]), dtype=np.float32))
            #self.X2 = tf.Variable(np.zeros(shape=(self.batch_size, self.training_data2.shape[1]), dtype=np.float32))

            #self.is_train = tf.placeholder(tf.bool, name="is_train")
            #self.is_train = tf.placeholder(tf.bool, name="is_train")
            #print("self.training_batch_size",self.batch_size)#16
            #print("self data1 size shape",self.training_data1.shape[1])#301
            #print("self data2 size shape",self.training_data2.shape[1])#301
            self.is_train = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool, name="is_train")
            #16 x 301
            #self.loss_ = self.loss(self.X1, self.X2)
            #print("loss self.X1",self.X1.shape)
            #print("loss self.X2",self.X2.shape)
            #print("X1 is",self.X1)
            #print("-------------------Good0-------------------------------")
            #print("The value of loss is :",self.loss_)
            #print("-------------------Good0 end-------------------------------")
            #if opt == 'Momentum':
                # c
                # self.train_step = tf.keras.optimizers.SGD(self.learning_rate, 0.9).minimize(self.loss_,tf.trainable_variables())
                #self.train_step = tf.keras.optimizers.SGD(self.learning_rate, 0.9).minimize(self.loss_,
                                                                                           # tf.compat.v1.trainable_variables())
              #  self.train_step = tf.keras.optimizers.SGD(self.learning_rate, 0.9)

               # print("--------------Good1---------------------")
            #else:
                # c
                # self.train_step = tf.keras.optimizers.SGD(self.learning_rate).minimize(self.loss_,tf.trainable_variables())
              #  self.train_step = tf.keras.optimizers.SGD(self.learning_rate)
              #  print("--------------Good2---------------------")

                # Initiate a tensor session
            #init = tf.global_variables_initializer()
            #self.sess = tf.Session()
            #self.sess.run(init)

            # train the model
            loss_cv, val_loss_cv = self.train(train_index, val_index)

            loss_cv += loss_cv
            val_loss_cv += val_loss_cv

        loss_cv = loss_cv / k
        val_loss_cv = val_loss_cv / k

        hist_costs = self.histcosts
        hist_val_costs = self.histvalcosts

        #self.sess.close()
        #tf.reset_default_graph()
        #del self.sess

        return {'loss': val_loss_cv, 'status': STATUS_OK, 'params': params, 'loss_train': loss_cv,
                'history_loss': hist_costs, 'history_val_loss': hist_val_costs}

    def train_test(self, params):
        batch_size = params['batch_size']
        self.n_hidden1 = params['units1']
        #print("Shape of n_hidden1 ", self.n_hidden1)
        self.n_hidden2 = params['units2']
        #print("Shape of n_hidden2 ", self.n_hidden2)
        self.alpha = params['alpha']
        self.lamda = params['lamda']
        self.learning_rate = params['learning_rate']

        learning_rate = params['learning_rate']

        init = params['initializer']
        if init == 'normal':
            self._init = tf.random_normal_initializer
        if init == 'uniform':
            self._init = tf.random_uniform_initializer
        if init == 'He':
            self._init = tf.keras.initializers.HeNormal()
        if init == 'xavier':
            self._init = tf.keras.initializers.GlorotNormal()

        opt = params['optimizer']
        if opt == 'SGD':
            #self.optimizer = tf.keras.optimizers.SGD()
            self.optimizer = tf.keras.optimizers.legacy.SGD()
        if opt == 'adam':
            #self.optimizer = tf.keras.optimizers.Adam()
            self.optimizer = tf.keras.optimizers.legacy.Adam()
        if opt == 'nadam':
            #self.optimizer = tf.keras.optimizers.Nadam()
            self.optimizer = tf.keras.optimizers.legacy.Nadam()
        if opt == 'Momentum':
            #self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate,momentum=0.9)
        if opt == 'RMSProp':
            #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
            self.optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=self.learning_rate)
        test_input1 = self.test_data1
        test_input2 = self.test_data2

        # tensor variables
        # X1 = tf.placeholder("float", shape=[None, self.test_data1.shape[1]])

        #X1 = tf.Variable(np.zeros((batch_size, self.test_data1.shape[1]), dtype=np.float32))

        # X2 = tf.placeholder("float", shape=[None, self.test_data2.shape[1]])

       # X2 = tf.Variable(np.zeros((batch_size, self.test_data2.shape[2]), dtype=np.float32))

        # self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.is_train = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool, name="is_train")

        # train the model
        #loss_ = self.loss(X1, X2)
        #if opt == 'Momentum':
            #train_step = tf.keras.optimizers.SGD(learning_rate, 0.9).minimize(loss_,
                                                                              #tf.compat.v1.trainable_variables())
            #train_step = tf.keras.optimizers.SGD(learning_rate, 0.9)


        #else:
            #train_step = tf.keras.optimizers.SGD(learning_rate).minimize(loss_,
                                                                        # tf.compat.v1.trainable_variables())
             #train_step = tf.keras.optimizers.SGD(learning_rate)
        # Initiate a tensor session
        #init = tf.global_variables_initializer()
        #self.sess = tf.Session()
       # self.sess.run(init)
       # save_sess = self.sess

        costs = []
        costs_inter = []

        epoch = 0
        best_cost = 100000
        stop = False
        n_samples = test_input1.shape[0]
        last_improvement = 0

        while epoch < self.max_epochs and stop == False:
            # train the model on the test set by mini batches
            # shuffle then split the test set into mini-batches of size self.batch_size
            seq = list(range(n_samples))
            random.shuffle(seq)
            mini_batches = [
                seq[k:k + batch_size]
                for k in range(0, n_samples, batch_size)
            ]
            avg_cost = 0.

            # Loop over all batches
            for sample in mini_batches:
                batch_xs1 = test_input1[sample][:]
                batch_xs2 = test_input2[sample][:]

                with tf.GradientTape() as tape:
                    # 计算当前批次的损失
                    current_loss = self.loss(batch_xs1, batch_xs2)

                    # 计算梯度
                gradients = tape.gradient(current_loss, self.trainable_variables)
                # print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
                # print(gradients)
                # print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
                # 更新模型的参数
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                cost = self.loss(batch_xs1, batch_xs2)
                avg_cost += cost[0] * len(sample) / n_samples

            costs_inter.append(avg_cost)

            # early stopping based on the validation data/ max_steps_without_decrease of the loss value : require_improvement
            if avg_cost < best_cost:
                #save_sess = self.sess
                best_cost = avg_cost
                costs += costs_inter
                last_improvement = 0
                costs_inter = []
            else:
                last_improvement += 1
            if last_improvement > self.require_improvement:
                # print("No improvement found in a while, stopping optimization.")
                # Break out from the loop.
                stop = True
                #self.sess = save_sess
            epoch += 1

        #self.sess.close()
        #tf.reset_default_graph()
        #del self.sess

        return best_cost


if __name__ == '__main__':
    selected_features = np.genfromtxt(
        r'C:\Users\gklizh\Documents\Workspace\code_and_data\data\python_related\data\selected_features.csv', delimiter=',',
        skip_header=1)

    # log10 (fpkm + 1)
    inputhtseq = np.genfromtxt(r'C:\Users\gklizh\Documents\Workspace\code_and_data\data\python_related\data\exp_intgr.csv',
                               dtype=np.unicode_, delimiter=',', skip_header=1)
    inputhtseq = inputhtseq[:, 1:inputhtseq.shape[1]].astype(float)
    inputhtseq = np.divide((inputhtseq - np.mean(inputhtseq)), np.std(inputhtseq))
    print(inputhtseq.shape)

    # methylation β values
    inputmethy = np.genfromtxt(r'C:\Users\gklizh\Documents\Workspace\code_and_data\data\python_related\data\mty_intgr.csv',
                               dtype=np.unicode_, delimiter=',', skip_header=1)
    inputmethy = inputmethy[:, 1:inputmethy.shape[1]].astype(float)
    inputmethy = np.divide((inputmethy - np.mean(inputmethy)), np.std(inputmethy))
    print(inputmethy.shape)

    # tanh activation function
    act = tf.nn.tanh

    # run the autoencoder for communities
    for iterator in range(1):
        print('iteration', iterator)
        selected_feat_cmt = selected_features[np.where(selected_features[:, 0] == iterator + 1)[0], :]

        print('first source ...')
        htseq_cmt = selected_feat_cmt[np.where(selected_feat_cmt[:, 1] == 1)[0], :]
        htseq_nbr = len(htseq_cmt)
        htseq_sel_data = inputhtseq[:, htseq_cmt[:, 2].astype(int) - 1]

        print("second source ...")
        methy_cmt = selected_feat_cmt[np.where(selected_feat_cmt[:, 1] == 2)[0], :]
        methy_nbr = len(methy_cmt)
        methy_sel_data = inputmethy[:, methy_cmt[:, 2].astype(int) - 1]

        print("features size of the 1st dataset:", htseq_nbr)
        print("features size of the 2nd dataset:", methy_nbr)

        n_hidden1 = htseq_nbr
        print("shape of n_hidden1 ", n_hidden1)
        n_hidden2 = methy_nbr
        print("shape of n_hidden2 ", n_hidden2)
        if htseq_nbr > 1 and methy_nbr > 1:
            # split dataset to training and test data 80%/20%
            X_train1, X_test1 = model_selection.train_test_split(htseq_sel_data, test_size=0.2, random_state=1)
            X_train2, X_test2 = model_selection.train_test_split(methy_sel_data, test_size=0.2, random_state=1)

            sae = AutoEncoder(X_train1, X_train2, X_test1, X_test2, activation=act)
            print("---------------------",X_train1.shape,X_train2.shape,X_test1.shape,X_test2.shape,"-------------------")


            trials = Trials()
            # define the space of hyper parameters
            space = {
                'units1': hp.choice('units1', range(1, n_hidden1)),
                'units2': hp.choice('units2', range(1, n_hidden2)),
                'batch_size': hp.choice('batch_size', [16, 8, 4]),
                'alpha': hp.choice('alpha', [0, hp.uniform('alpha2', 0, 1)]),
                'learning_rate': hp.loguniform('learning_rate', -5, -1),
                'lamda': hp.choice('lamda', [0, hp.loguniform('lamda2', -8, -1)]),
                'optimizer': hp.choice('optimizer', ["adam", "nadam", "SGD", "Momentum", "RMSProp"]),
                'initializer': hp.choice('initializer', ["xavier"]),
            }

            # train the HP optimization with 20 iterations
            best = fmin(sae.cross_validation, space, algo=tpe.suggest, max_evals=20, trials=trials)
            print(best)

            # save the best HPs
            fname = (r'C:\Users\gklizh\Documents\Workspace\code_and_data\data\python_related\result\comm_trials_binary'
                     r'.pkl')
            pickle.dump(trials, open(fname, "ab"))

            # get the loss of training the model on test data
            loss = sae.train_test(hyperopt.space_eval(space, best))

            f = open(r'C:\Users\gklizh\Documents\Workspace\code_and_data\data\python_related\result\comm_parameter_binary.txt', 'a+')
            for k, v in best.items():
                f.write(str(k) + ':' + str(v) + ',')
            f.write(str(loss))
            f.write('\n')
            f.close()
            # Release memory
            del htseq_sel_data
            del methy_sel_data
            del trials
            del sae
