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
import time
from datetime import datetime
import logging
import datetime

# 获取当前时间并格式化为字符串，例如 '2024-01-18'
current_time = datetime.datetime.now().strftime("%Y-%m-%d")

# 构造带有时间戳的文件名
log_filename = f'myapp.{current_time}_test14.log'

# 配置日志
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(message)s')
warnings.filterwarnings('ignore')


# from tensorflow.python.training.momentum import MomentumOptimizer


class AutoEncoder(tf.keras.Model):

    def __init__(self, data1, data2, testdata1=None, testdata2=None, n_hiddensh=1, activation=None):

        __slots__ = ['data1', 'data2','testdata1','testdata2','n_hiddensh','activation']

        super(AutoEncoder, self).__init__()
        # training datasets
        self.crosser = None
        self.spliter = 0
        self.iterator = 0
        self.cross_counter = 0
        self.n_layer5 = None
        self.histcosts = None
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
        # print("shape of n_hiddensh", n_hiddensh)
        self.activation = activation
        self.is_train = False

    def encode(self, X1, X2):
        # =============================================================================
        # first hidden layer composed of two parts related to two sources (X1, X2)
        # - build a fully connected layer
        # - apply the batch normalization
        # - apply an activation function
        # =============================================================================
        self.l1_layer = tf.keras.layers.Dense(self.n_hidden1, kernel_initializer=self._init, name='layer1')
        l1 = self.l1_layer(X1)

        l1 = tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))
        l1 = self.activation(l1)
        self.n_layer1 = l1.shape
        self.l1 = l1
        # print("layer1 shape", l1.shape)
        self.l2_layer = tf.keras.layers.Dense(self.n_hidden2, kernel_initializer=self._init, name='layer2')
        l2 = self.l2_layer(X2)
        l2 = tf.keras.layers.BatchNormalization()(l2, training=bool(self.is_train))
        l2 = self.activation(l2)
        self.n_layer2 = l2.shape
        self.l2 = l2
        # print("layer2 shape", l2.shape)
        # =============================================================================
        # fuse the parts of the first hidden layer
        # =============================================================================
        self.l3_layer = tf.keras.layers.Dense(self.n_hiddensh, kernel_initializer=self._init, name='layer3')
        l3 = self.l3_layer(tf.concat([l1, l2], 1))
        # l1,l2 are bounded in l
        # print("-----layer3 shape", l3.shape)
        l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))
        self.n_layer3 = l3.shape
        self.l3 = l3
        # print("layer3 shape", l3.shape)
        return self.l3  # if we can use self.l3

    def decode(self, H):
        self.l4_layer = tf.keras.layers.Dense(self.n_hidden1 + self.n_hidden2, kernel_initializer=self._init,
                                              name='layer4')
        l4 = self.l4_layer(H)
        l4 = tf.keras.layers.BatchNormalization()(l4, training=bool(self.is_train))
        self.n_layer4 = l4.shape
        self.l4 = l4
        # print("layer4 shape", l4.shape)
        s1, s2 = tf.split(l4, [self.n_hidden1, self.n_hidden2], 1)
        self.l5_layer = tf.keras.layers.Dense(self.n_input1, kernel_initializer=self._init, name='layer5')
        l5 = self.l5_layer(s1)
        l5 = tf.keras.layers.BatchNormalization()(l5, training=bool(self.is_train))
        l5 = self.activation(l5)  # l1
        self.n_layer5 = l5.shape
        self.l5 = l5
        # print("layer5 shape", l5.shape)
        self.l6_layer = tf.keras.layers.Dense(self.n_input2, kernel_initializer=self._init, name='layer6')
        l6 = self.l6_layer(s2)
        l6 = tf.keras.layers.BatchNormalization()(l6, training=bool(self.is_train))
        l6 = self.activation(l6)  # l2
        self.n_layer6 = l6.shape
        self.l6 = l6
        # print("layer6 shape", l6.shape)
        return self.l5, self.l6

    # def get_weights(self):
    # self.W1 = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=self.n_layer1), trainable=True,
    #                       name="layer1/kernel")
    # self.W2 = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=self.n_layer2), trainable=True,
    #                       name="layer2/kernel")
    # self.Wsh = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=self.n_layer3), trainable=True,
    #                        name="layer3/kernel")
    # self.Wsht = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=self.n_layer4), trainable=True,
    #                        name="layer4/kernel")
    #  self.W1t = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=self.n_layer5), trainable=True,
    #                         name="layer5/kernel")
    #   self.W2t = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal()(shape=self.n_layer6), trainable=True,
    #    #                      name="layer6/kernel")

    def encodefun(self, X1, X2):

        l1 = self.l1_layer(X1)
        l1 = tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))
        l1 = self.activation(l1)
        self.n_layer1 = l1.shape
        self.l1 = l1

        l2 = self.l2_layer(X2)
        l2 = tf.keras.layers.BatchNormalization()(l2, training=bool(self.is_train))
        l2 = self.activation(l2)
        self.n_layer2 = l2.shape
        self.l2 = l2
        l3 = self.l3_layer(tf.concat([l1, l2], 1))
        # l1,l2 are bounded in l
        # print("-----layer3 shape", l3.shape)
        l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))
        self.n_layer3 = l3.shape
        self.l3 = l3
        # print("layer3 shape", l3.shape)
        return self.l3

    def decodefun(self, H):

        l4 = self.l4_layer(H)
        l4 = tf.keras.layers.BatchNormalization()(l4, training=bool(self.is_train))
        self.n_layer4 = l4.shape
        self.l4 = l4
        s1, s2 = tf.split(l4, [self.n_hidden1, self.n_hidden2], 1)

        l5 = self.l5_layer(s1)
        l5 = tf.keras.layers.BatchNormalization()(l5, training=bool(self.is_train))
        l5 = self.activation(l5)  # l1
        self.n_layer5 = l5.shape
        self.l5 = l5

        l6 = self.l6_layer(s2)
        l6 = tf.keras.layers.BatchNormalization()(l6, training=bool(self.is_train))
        l6 = self.activation(l6)  # l2
        self.n_layer6 = l6.shape
        self.l6 = l6
        return self.l5, self.l6

    def lossfun(self, X1, X2):

        self.H = self.encodefun(X1, X2)
        X1_, X2_ = self.decodefun(self.H)
        self.get_weights()
        sgroup_lasso = self.L2regularization(self.W1, self.n_input1 * self.n_hidden1) + \
                       self.L2regularization(self.W2, self.n_input2 * self.n_hidden2)
        #print(sgroup_lasso.shape)
        # lasso
        lasso = self.L1regularization(self.W1) + self.L1regularization(self.W2) + \
                self.L1regularization(self.Wsh) + self.L1regularization(self.Wsht) + \
                self.L1regularization(self.W1t) + self.L1regularization(self.W2t)
        #print(lasso.shape)
        error = tf.reduce_mean(tf.square(X1 - X1_)) + tf.reduce_mean(tf.square(X2 - X2_))

        #print(error.shape)

        #error = tf.cast(error, tf.float32)  # 或 tf.float64，根据需要
        #sgroup_lasso = tf.cast(sgroup_lasso, tf.float32)  # 同上
        #lasso = tf.cast(lasso, tf.float32)
        #print(f"loss fun error:{error}")
        cost = 0.5 * error + 0.5 * self.lamda * (1 - self.alpha) * sgroup_lasso + 0.5 * self.lamda * self.alpha * lasso
        return cost

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
        #print(sgroup_lasso.shape)
        # lasso
        lasso = self.L1regularization(self.W1) + self.L1regularization(self.W2) + \
                self.L1regularization(self.Wsh) + self.L1regularization(self.Wsht) + \
                self.L1regularization(self.W1t) + self.L1regularization(self.W2t)

        # reconstruction Error
        # print("------check the value of weights in patches-------------------")
        # print("------------------------W1-------------------")
        # print(self.W1)
        # print("--------------------self.kernel------------------------")
        # print(self.l1_layer.kernel)
        # print("------------------------weights in patches end-------------------")
        #print(lasso.shape)
        #error = tf.reduce_sum(tf.losses.mean_squared_error(X1, X1_)) + tf.reduce_sum(tf.losses.mean_squared_error(X2, X2_))
        error = tf.reduce_mean(tf.square(X1 - X1_)) + tf.reduce_mean(tf.square(X2 - X2_))
        #print(error.shape)
        #error = tf.cast(error, tf.float32)  # 或 tf.float64，根据需要
        #sgroup_lasso = tf.cast(sgroup_lasso, tf.float32)  # 同上
        #lasso = tf.cast(lasso, tf.float32)
        #print(f'X1 is:{X1.shape}')
        #print(f'X1 is:{X1_.shape}')
        #print(f'X1 is:{X2.shape}')
        #print(f'X1 is:{X2_.shape}')
        #print(f"error:{error}")
        # Loss function
        cost = 0.5 * error + 0.5 * self.lamda * (1 - self.alpha) * sgroup_lasso + 0.5 * self.lamda * self.alpha * lasso
        return cost

    def train(self, train_index, val_index):
        # training data
        train_input1 = self.training_data1[train_index, :]  # 370 x 4/5
        train_input2 = self.training_data2[train_index, :]
        # print("The size of  train_input1",self.training_data1.shape)
        logging.info("---------------TRain starts---------")
        # validation data
        val_input1 = self.training_data1[val_index, :]  # 370 x 1/5
        val_input2 = self.training_data2[val_index, :]

        # save_sess = self.sess

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
        # print("n_samples is :",n_samples)
        # print("vn_samples is :",vn_samples)
        # print("self.batch_size is :",self.batch_size)
        # train the mini_batches model using the early stopping criteria
        # for var in self.trainable_variables:
        # print("The name of parameter and shape", var.name, var.shape)

        epoch = 0
        counter = 0
        while epoch < self.max_epochs and stop == False:
            # for(self.max)
            # train the model on the training set by mini batches
            # shuffle then split the training set to mini-batches of size self.batch_size
            logging.info(
                f"#################################################epoch :{epoch}#################################################")
            seq = list(range(n_samples))  # 370 x 4/5
            # print("The number of n_samples",n_samples)#370 x 4/5
            random.shuffle(seq)
            mini_batches = [
                seq[k:k + self.batch_size]
                for k in range(0, n_samples, self.batch_size)
            ]

            avg_cost = 0.  # the average cost of mini_batches
            avg_cost_val = 0.
            logging.info(
                "----------------------one trial for train samples starts one epoch ------------------------\n")
            for sample in mini_batches:
                # print("#############Sample:", len(sample))
                batch_xs1 = train_input1[sample][:]
                batch_xs2 = train_input2[sample][:]
                self.is_train = True
                # feed_dictio = {self.X1: batch_xs1, self.X2: batch_xs2, self.is_train: True}
                # cost = self.sess.run([self.loss_, self.train_step], feed_dict=feed_dictio)
                # avg_cost += cost[0] * len(sample) / n_samples

                with tf.GradientTape() as tape:
                    # 计算当前批次的损失
                    if epoch == 0 and counter == 0:
                        current_loss = self.loss(batch_xs1, batch_xs2)
                    else:
                        current_loss = self.lossfun(batch_xs1, batch_xs2)
                logging.info(
                    f"----------------------check if the weighths  added into tape loss in this patch -----{epoch}-------------------------")
                logging.info(self.trainable_variables)
                logging.info(
                    "-----------------------------------------patch ends in tape loss -----------------------------------------")
                # 计算梯度
                gradients = tape.gradient(current_loss, self.trainable_variables)
                # print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
                # print(gradients)
                # print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
                # 更新模型的参数
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                logging.info(
                    f"----------------------check if the weighths has been approved in this patch -----{epoch}-------------------------")
                logging.info(self.trainable_variables)
                logging.info("-----------------------------------------patch ends in train-----------------------------------------")
                cost = self.lossfun(batch_xs1, batch_xs2)
                # print("---------------------train costs-------------")
                # print("cost:",cost)

                # print("-----------------------Kosten---------------------------",cost)
                # print("---------------------------------------------")
                avg_cost += cost * len(sample) / n_samples
                counter += 1
                # print("----------------------------cost_train:", cost, avg_cost, "-----------------------------")
            # print("---------------------train costs ends-------------")

            # train the model on the validation set by mini batches
            # split the validation set to mini-batches of size self.batch_size
            logging.info("----------------------one trial for to train samples ends one for ------------------------")
            logging.info(self.trainable_variables)
            logging.info("---------------------------train sample ends one for------------------------------")
            # -------------------------------------------------------------------#
            # print("-----------------check after training the kernel value train for ends----------------------")

            # print("-----------------for ends check---------------------")
            seq = list(range(vn_samples))
            mini_batches = [
                seq[k:k + self.batch_size]
                for k in range(0, vn_samples, self.batch_size)
            ]
            # avg_cost_val = 0.
            counter = 0
            logging.info("--------------------------The validation trial starts------------------------------------")
            for sample in mini_batches:
                batch_xs1 = val_input1[sample][:]
                batch_xs2 = val_input2[sample][:]
                self.is_train = False
                # print("#############Sample:",len(sample))
                # feed_dictio = {self.X1: batch_xs1, self.X2: batch_xs2, self.is_train: False}
                cost_val = self.lossfun(batch_xs1, batch_xs2)
                logging.info(self.trainable_variables)
                # self.lossfun(batch_xs1, batch_xs2)
                # print("---------------cost[0] val territory")

                # print(cost_val)
                # print("---------------cost[0] val territory ends")
                avg_cost_val += cost_val * len(sample) / vn_samples
                # print("----------------------------cost_val:", cost_val,avg_cost_val, "-----------------------------")
            # cost history since the last best cost

            logging.info("++++++++++++++++++++++++The validation trial ends-++++++++++++++++++++++++++++++++++\n")
            # logging.info("\n")
            costs_inter.append(avg_cost)
            costs_val_inter.append(avg_cost_val)

            # print("----------------------------------------------costs_val_inter--------------------------------------")
            # print("set of avg_cost_val from costs_val_inter", costs_val_inter)
            # print("costs_val_interp[0]", costs_val_inter[0])
            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print("set of avg_cost from costs_inter:", costs_inter)
            # print("costs_inter[0]:", costs_inter[0])
            # print("----------------------------------------------costs_val_inter--------------------------------------")
            # early stopping based on the validation set/ max_steps_without_decrease of the loss value: require_improvement
            # print("--------------In if region---------------------")

            # print(f"Before if avg_cost_val:{avg_cost_val}, best_val_cost:{best_val_cost}")
            if avg_cost_val < best_val_cost:
                # print("avg_cost_val is :",avg_cost_val)
                # save_sess = self.sess  # save session
                best_val_cost = avg_cost_val
                # print("###########################################")
                #  print("show me the costs_val_inter",costs_val_inter)
                # print("show me the costs_inter", costs_inter)
                best_cost = avg_cost
                costs_val += costs_val_inter  # costs history of the validation set
                costs += costs_inter  # costs history of the training set

                # print("show me the costs",costs)

                # print("show me the costs_val",costs_val)
                # print("###########################################")
                last_improvement = 0
                costs_val_inter = []
                costs_inter = []

            else:
                last_improvement += 1
                # costs_val += costs_val_inter  # costs history of the validation set
                # costs += costs_inter
            # costs_val_inter = []
            # costs_inter = []
            # print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~This is else part and we will see the value of avg_cost_val :{avg_cost_val} and best_val_cost:{best_val_cost}~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if last_improvement > self.require_improvement:
                # print("No improvement found during the (self.require_improvement) last iterations, stopping optimization.")
                # Break out from the loop.
                stop = True
                # self.sess = save_sess  # restore session with the best cost

            epoch += 1
        # print("++++++++++++++++++++++++++The end of while++++++++++++++++++++++++++++++++++++++")
        # self.histcosts = costs
        # self.histvalcosts = costs_val
        self.histcosts = ([float(tensor_histcosts.numpy()) for tensor_histcosts in costs])
        # self.histcosts = list(self.histcosts)
        # self.histcosts = [scaler_v1 for scaler_v1 in self.histcosts]

        self.histvalcosts = ([float(tensor_histvalcosts.numpy()) for tensor_histvalcosts in costs_val])
        # self.histvalcosts = list(self.histvalcosts )
        # self.histvalcosts = [scaler_v2 for scaler_v2 in self.histvalcosts]

        logging.info("---------------End test------------------")
        logging.info(f"The value of histcosts:{self.histcosts}")
        # for display in self.histcosts:
        # print(display.numpy)
        logging.info("\nThe value of histvalcosts:{self.histvalcosts}")
        # for displayval in self.histvalcosts:
        # print(displayval.numpy)
        logging.info("---------------End test in history displaying----------")
        counter = 0
        epoch =0
        plt.figure()
        plt.plot(costs)
        plt.ylabel('cost Loss')
        plt.xlabel('Iterations')
        plt.title("Learning rate =" + str(round(self.learning_rate, 3)))
        plt.savefig(r'C:\Users\gklizh\Documents\Workspace\code_and_data15\figure\loss_curve\training_picture' + str(iterator) + str('_')+str(self.spliter) + '_test19.png')
        plt.close()

        plt.figure()
        plt.plot(costs_val)
        plt.ylabel('validation Loss')
        plt.xlabel('Iterations')
        plt.title("Learning rate =" + str(round(self.learning_rate, 3)))
        plt.savefig(r'C:\Users\gklizh\Documents\Workspace\code_and_data15\figure\loss_curve\validation_picture' + str(iterator) +str('_')+str(self.spliter) + '_test19.png')
        plt.close()
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
        logging.info("------------------------------Cross validation starts-------------------------------------")
        self.batch_size = params['batch_size']
        # print("self.batch_size",self.batch_size)#16
        self.n_hidden1 = params['units1']
        self.n_hidden2 = params['units2']
        self.alpha = params['alpha']
        self.lamda = params['lamda']
        self.learning_rate = params['learning_rate']
        self.cross_counter = 0
        # k fold validation
        k = 5
        self.require_improvement = 20
        self.max_epochs = 1000
        init = params['initializer']
        if init == 'normal':
            self._init = tf.keras.initializers.RandomNormal()
        if init == 'uniform':
            self._init = tf.keras.initializers.RandomUniform()
        if init == 'He':
            self._init = tf.keras.initializers.HeNormal()
        if init == 'xavier':
            self._init = tf.keras.initializers.GlorotNormal()

        opt = params['optimizer']
        # if opt == 'SGD':
        # self.optimizer = tf.keras.optimizers.SGD()
        # if opt == 'adam':
        #    self.optimizer = tf.keras.optimizers.Adam()
        # if opt == 'nadam':
        #    self.optimizer = tf.keras.optimizers.Nadam()
        # if opt == 'Momentum':
        #     self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate,momentum=0.9)
        # if opt == 'RMSProp':
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
            self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate, momentum=0.9)
        if opt == 'RMSProp':
            # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
            self.optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=self.learning_rate)

        # cross-validation
        data = np.concatenate([self.training_data1, self.training_data2], axis=1)
        # print("data shape is ",data.shape)
        kf = KFold(n_splits=k, shuffle=True)  # k fold cross validation
        kf.get_n_splits(data)  # returns the number of splitting iterations in the cross-validator
        # data is component with 301+301
        # validation set loss
        loss_cv = 0
        val_loss_cv = 0
        # counter=0
        # print("train check self.trainable_variables()",self.trainable_variables())
        self.spliter = 0
        for train_index, val_index in kf.split(data):
            # counter+=1
            logging.info(f"train_index:{train_index.shape}")
            logging.info(f"val_index:{val_index.shape}")
            #self.is_train = True
            # reset tensor graph after each cross_validation run
            # tf.reset_default_graph()
            # model = AutoEncoder(self.training_data1, self.training_data2, self.test_data1, self.test_data2,
            #      self.n_hiddensh,
            #      self.activation)
            # self.X1 = tf.placeholder("float", shape=[None, self.training_data1.shape[1]])
            # self.X2 = tf.placeholder("float", shape=[None, self.training_data2.shape[1]])

            # self.X1 = tf.Variable(np.zeros(shape=(self.batch_size, self.training_data1.shape[1]), dtype=np.float32))
            # self.X2 = tf.Variable(np.zeros(shape=(self.batch_size, self.training_data2.shape[1]), dtype=np.float32))

            # self.is_train = tf.placeholder(tf.bool, name="is_train")
            # self.is_train = tf.placeholder(tf.bool, name="is_train")
            # print("self.training_batch_size",self.batch_size)#16
            # print("self data1 size shape",self.training_data1.shape[1])#301
            # print("self data2 size shape",self.training_data2.shape[1])#301
            #self.is_train = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool, name="is_train")
            # 16 x 301
            # self.loss_ = self.loss(self.X1, self.X2)
            # print("loss self.X1",self.X1.shape)
            # print("loss self.X2",self.X2.shape)
            # print("X1 is",self.X1)
            # print("-------------------Good0-------------------------------")
            # print("The value of loss is :",self.loss_)
            # print("-------------------Good0 end-------------------------------")
            # if opt == 'Momentum':
            # c
            # self.train_step = tf.keras.optimizers.SGD(self.learning_rate, 0.9).minimize(self.loss_,tf.trainable_variables())
            # self.train_step = tf.keras.optimizers.SGD(self.learning_rate, 0.9).minimize(self.loss_,
            # tf.compat.v1.trainable_variables())
            #  self.train_step = tf.keras.optimizers.SGD(self.learning_rate, 0.9)

            # print("--------------Good1---------------------")
            # else:
            # c
            # self.train_step = tf.keras.optimizers.SGD(self.learning_rate).minimize(self.loss_,tf.trainable_variables())
            #  self.train_step = tf.keras.optimizers.SGD(self.learning_rate)
            #  print("--------------Good2---------------------")

            # Initiate a tensor session
            # init = tf.global_variables_initializer()
            # self.sess = tf.Session()
            # self.sess.run(init)

            # train the model
            logging.info(
                "-----------------------One iteration in for train_index, val_index in kf.split(data)-----------------------------")

            loss_cv, val_loss_cv = self.train(train_index, val_index)
            self.spliter += 1
            #self.cross_counter += 1
            # best_train,best_val
            loss_cv += loss_cv
            val_loss_cv += val_loss_cv

        loss_cv = loss_cv / k
        val_loss_cv = val_loss_cv / k
        self.cross_counter = 0
        hist_costs = self.histcosts
        hist_val_costs = self.histvalcosts
        logging.info("************************The crossvalidation ->hiscosts:*********************")
        logging.info(hist_costs)
        logging.info("************************The crossvalidation ->hist_val_costs:*********************")
        logging.info(hist_val_costs)

        # self.sess.close()
        # tf.reset_default_graph()
        # del self.sess
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%trian%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print(self.histcosts)
        # print(self.histvalcosts)
        # print(val_loss_cv)
        # print(STATUS_OK)
        # print(loss_cv)
        # print(hist_costs)
        # print(hist_val_costs)
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%trian%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logging.info("--------------------------Cross validation ends-----------------------------------------")
        return {'loss': val_loss_cv, 'status': STATUS_OK, 'params': params, 'loss_train': loss_cv,
                'history_loss': hist_costs, 'history_val_loss': hist_val_costs}

    def train_test(self, params):
        logging.info("------------test train starts--------------------")
        batch_size = params['batch_size']
        self.n_hidden1 = params['units1']
        # print("Shape of n_hidden1 ", self.n_hidden1)
        self.n_hidden2 = params['units2']
        # print("Shape of n_hidden2 ", self.n_hidden2)
        self.alpha = params['alpha']
        self.lamda = params['lamda']
        self.learning_rate = params['learning_rate']

        learning_rate = params['learning_rate']

        init = params['initializer']
        if init == 'normal':
            self._init = tf.keras.initializers.RandomNormal()
        if init == 'uniform':
            self._init = tf.keras.initializers.RandomUniform()
        if init == 'He':
            self._init = tf.keras.initializers.HeNormal()
        if init == 'xavier':
            self._init = tf.keras.initializers.GlorotNormal()

        opt = params['optimizer']
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
            self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate, momentum=0.9)
        if opt == 'RMSProp':
            # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
            self.optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=self.learning_rate)
        test_input1 = self.test_data1
        test_input2 = self.test_data2

        # tensor variables
        # X1 = tf.placeholder("float", shape=[None, self.test_data1.shape[1]])

        # X1 = tf.Variable(np.zeros((batch_size, self.test_data1.shape[1]), dtype=np.float32))

        # X2 = tf.placeholder("float", shape=[None, self.test_data2.shape[1]])

        # X2 = tf.Variable(np.zeros((batch_size, self.test_data2.shape[2]), dtype=np.float32))

        # self.is_train = tf.placeholder(tf.bool, name="is_train")
        #self.is_train = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool, name="is_train")

        # train the model
        # loss_ = self.loss(X1, X2)
        # if opt == 'Momentum':
        # train_step = tf.keras.optimizers.SGD(learning_rate, 0.9).minimize(loss_,
        # tf.compat.v1.trainable_variables())
        # train_step = tf.keras.optimizers.SGD(learning_rate, 0.9)

        # else:
        # train_step = tf.keras.optimizers.SGD(learning_rate).minimize(loss_,
        # tf.compat.v1.trainable_variables())
        # train_step = tf.keras.optimizers.SGD(learning_rate)
        # Initiate a tensor session
        # init = tf.global_variables_initializer()
        # self.sess = tf.Session()
        # self.sess.run(init)
        # save_sess = self.sess

        costs = []
        costs_inter = []

        epoch = 0
        counter = 0
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
                self.is_train=True
                #logging.info(batch_xs1.shape)
                #logging.info(batch_xs2.shape)
                with tf.GradientTape() as tape:
                    # 计算当前批次的损失
                    #current_loss = self.lossfun(batch_xs1, batch_xs2)
                    if epoch == 0 and counter == 0:
                        current_loss = self.loss(batch_xs1, batch_xs2)
                    else:
                        current_loss = self.lossfun(batch_xs1, batch_xs2)

                    # 计算梯度
                gradients = tape.gradient(current_loss, self.trainable_variables)
                # print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
                # print(gradients)
                # print("~~~~~~~~~~~~~~~~~~~~value exists~~~~~~~~~~~~~~~~")
                # 更新模型的参数
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                # print("--------------------------train test is -------------------------------:",
                #  self.trainable_variables)
                cost = self.lossfun(batch_xs1, batch_xs2)
                avg_cost += cost * len(sample) / n_samples
                counter += 1

            costs_inter.append(avg_cost)

            # early stopping based on the validation data/ max_steps_without_decrease of the loss value : require_improvement
            if avg_cost < best_cost:
                # save_sess = self.sess
                best_cost = avg_cost
                costs += costs_inter
                last_improvement = 0
                costs_inter = []
            else:
                last_improvement += 1
                # costs += costs_inter
                # costs_inter = []
            # costs_inter = []
            if last_improvement > self.require_improvement:
                # print("No improvement found in a while, stopping optimization.")
                # Break out from the loop.
                stop = True
                # self.sess = save_sess
            epoch += 1
        # costs = list([tensor_costs for tensor_costs in costs])
        # costs = [scaler_v3 for scaler_v3 in costs ]
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^The end of train_test^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # print("The costs in train test is ", costs)
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # self.sess.close()
        # tf.reset_default_graph()
        # del self.sess
        plt.figure()
        plt.plot(costs)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.title("Learning rate =" + str(round(self.learning_rate, 3)))
        plt.savefig(r'C:\Users\gklizh\Documents\Workspace\code_and_data15\figure\loss_curve\train_test_picture_results' + str(iterator) + '_test19.png')
        plt.close()
        return best_cost


if __name__ == '__main__':

    f = open(
        r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\result\comm_parameter_binary19.txt',
        'w+')
    f.close()

    fname = r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\result\comm_trials_binary_test19.pkl'
    with open(fname, 'wb+') as fpkl:
        pass
    # with open(fname, "wb") as f:
    # pass

    start_time = time.perf_counter()
    selected_features = np.genfromtxt(
        r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\data\selected_features.csv',
        delimiter=',',
        skip_header=1)

    # log10 (fpkm + 1)
    inputhtseq = np.genfromtxt(
        r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\data\exp_intgr.csv',
        dtype=np.unicode_, delimiter=',', skip_header=1)
    inputhtseq = inputhtseq[:, 1:inputhtseq.shape[1]].astype(float)
    inputhtseq = np.divide((inputhtseq - np.mean(inputhtseq)), np.std(inputhtseq))
    print(inputhtseq.shape)

    # methylation β values
    inputmethy = np.genfromtxt(
        r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\data\mty_intgr.csv',
        dtype=np.unicode_, delimiter=',', skip_header=1)
    inputmethy = inputmethy[:, 1:inputmethy.shape[1]].astype(float)
    inputmethy = np.divide((inputmethy - np.mean(inputmethy)), np.std(inputmethy))
    print(inputmethy.shape)

    # tanh activation function
    act = tf.nn.tanh

    # run the autoencoder for communities
    trials = {}
    for iterator in range(21):
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
            #sae.iterator = iterator
            # print("---------------------",X_train1.shape,X_train2.shape,X_test1.shape,X_test2.shape, "-------------------")
            # trialsx = Trials()
            trial_label = f"trial_{iterator}"  # 创建标签，例如 "trial_1"
            trials[trial_label] = Trials()
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

            logging.info("---------Start trial----------------")
            best = fmin(sae.cross_validation, space, algo=tpe.suggest, max_evals=2, trials=trials[trial_label])
            #sae.crosser += 1

            logging.info("---------ends trial----------------")
            with open(fname, "ab") as file:
                pickle.dump(trials[trial_label], file)
            # print("------------------Got the best value----------")
            # print(best)
            # print("-----------------------End output best------------")
            # save the best HPs
            # fname = (r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\result\comm_trials_binary_test06.pkl')
            # pickle.dump(trialsx, open(fname, "ab"))
            # with open(fname, "ab") as file:
            # 将 trials 对象序列化并写入到文件中
            # pickle.dump(trialsx, file)
            # get the loss of training the model on test data
            loss = sae.train_test(hyperopt.space_eval(space, best))

            # print("-------------------end the train test-------------------------")
            # print("The loss value :", loss)
            # print("-------------------end the loss display-------------------------")
            # print("-------------display best items----------------------")

            # print(best.items())

            # print("-------------end best items----------------------")

            # print("-------------display best ----------------------")

            # print(best)
            # print(best.items())

            # print("-------------end best ---------------------")

            f = open(
                r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\result\comm_parameter_binary19.txt',
                'a+')
            # print("---------open the file to load the best items in best-----------------------------")

            for k, v in best.items():
                f.write(str(k) + ':' + str(v) + ',')
                # print("k",str(k))
                # print("v",str(v))
            # print("The file f recording ends")
            f.write("---------------------------------------------")
            f.write(str(loss))
            f.write('\n')
            f.close()

            end_time = time.perf_counter()
            # print("-------------Time count--------------------")
            # print(f"The total time it took is about {end_time - start_time} seconds")
            # print("-------------Time count--------------------")
            # f1 = open(r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\result\comm_parameter_binary_backup.txt',
            #     'a+')
            # print("---------open the file to load the best items in best-----------------------------")
            # for k1, v1 in best.items():
            #    f1.write(str(k1) + ':' + str(v1) + ',')
            # print("k", str(k1))
            # print("v", str(v1))
            # print("The file f1 recording ends")
            # f1.write('\n')
            # f1.close()
            # Release memory
            del htseq_sel_data
            del methy_sel_data
            del sae
            # del trialsx
    # print("-----Trails record history--------")
    # print(trials.items())
    # print("----------------------------------")

    # for trial_label, trial in trials.items():
    # print(f"\nData for {trial_label}:")
    # with open(fname, "ab") as file:
    # pickle.dump(trials[trial_label], file)
    # pickle.dump(trials[trial_label], open(fname, "ab"))

    for trial_label, trial in trials.items():
        print(f"\nData for {trial_label}:")
        for trial_result in trial.trials:
            print(trial_result)
