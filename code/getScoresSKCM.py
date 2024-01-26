# coding = utf-8

import tensorflow as tf
import pickle
import math
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import shap
import warnings
import time
print(np.__version__)

class AutoEncoder(tf.keras.Model):

    def __init__(self, data1, data2, n_hiddensh=1, activation=None):
        super(AutoEncoder, self).__init__()
        # training datasets
        #self.trainable_variables = None
        #self.trainable_variables = None
        self.model_H = None
        self._init = None
        self.mainer=0
        self.H = None
        self.batch_xs2 = None
        self.W2t = None
        self.Wsht = None
        self.W1t = None
        self.Wsh = None
        self.W2 = None
        self.W1 = None
        self.n_layer6 = None
        self.l6_layer = None
        self.n_layer5 = None
        self.l5_layer = None
        self.l4_layer = None
        self.n_layer3 = None
        self.n_layer2 = None
        self.l2_layer = None
        self.n_layer1 = None
        self.l1_layer = None
        self.l3_layer = None
        self.n_layer4 = None
        self.batch_xs1 = None
        self.l5 = None
        self.l4 = None
        self.l3 = None
        self.l6 = None
        self.l2 = None
        self.l1 = None
        self.training_data1 = data1
        self.training_data2 = data2
        # test datasets
        #self.test_data1 = testdata1
        #self.test_data2 = testdata2
        # number of features
        self.n_input1 = data1.shape[1]
        self.n_input2 = data2.shape[1]
        self.n_hiddensh = n_hiddensh
        # activation function
        #print("shape of n_hiddensh", n_hiddensh)
        self.activation = activation
        #self.encoder_model = self.modelCreate()
        self.input_layer1=None
        self.input_layer2=None
        #self.trainable_variables = []

    def encode(self, X1, X2):
        # =============================================================================
        # first hidden layer composed of two parts related to two sources (X1, X2)
        # - build a fully connected layer
        # - apply the batch normalization
        # - apply an activation function
        # =============================================================================
        #self.input_layer1 = X1
        #self.input_layer2 = X2
        self.l1_layer = tf.keras.layers.Dense(self.n_hidden1, kernel_initializer=self._init, name='layer1')
        l1 = self.l1_layer(X1)

        l1 = tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))
        l1 = self.activation(l1)
        self.n_layer1 = l1.shape
        self.l1 = l1
        #self.trainable_variables.append(self.l1_layer.kernel)
        #self.trainable_variables.append(self.l1_layer.bias)
        # print("layer1 shape", l1.shape)
        self.l2_layer = tf.keras.layers.Dense(self.n_hidden2, kernel_initializer=self._init, name='layer2')
        l2 = self.l2_layer(X2)
        l2 = tf.keras.layers.BatchNormalization()(l2, training=bool(self.is_train))
        l2 = self.activation(l2)
        self.n_layer2 = l2.shape
        self.l2 = l2
        #self.trainable_variables.append(self.l2_layer.kernel)
        #self.trainable_variables.append(self.l2_layer.bias)
        # print("layer2 shape", l2.shape)
        # =============================================================================
        # fuse the parts of the first hidden layer
        # =============================================================================
        #self.trainable_variables.append(self.l2_layer.kernel)
        self.l3_layer = tf.keras.layers.Dense(self.n_hiddensh, kernel_initializer=self._init, name='layer3')
        l3 = self.l3_layer(tf.concat([l1, l2], 1))
        # l1,l2 are bounded in l
        # print("-----layer3 shape", l3.shape)
        l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))
        self.n_layer3 = l3.shape
        self.l3 = l3
        #self.trainable_variables.append(self.l3_layer.kernel)
        #self.trainable_variables.append(self.l3_layer.bias)
        # print("layer3 shape", l3.shape)
       # self.trainable_variables.append(self.l3_layer.kernel)

        return l3

    def decode(self, H):
        self.l4_layer = tf.keras.layers.Dense(self.n_hidden1 + self.n_hidden2, kernel_initializer=self._init,
                                              name='layer4')
        l4 = self.l4_layer(H)
        l4 = tf.keras.layers.BatchNormalization()(l4, training=bool(self.is_train))
        self.n_layer4 = l4.shape
        self.l4 = l4
        #self.trainable_variables.append(self.l4_layer.kernel)
        #self.trainable_variables.append(self.l4_layer.bias)
       # self.trainable_variables.append(self.l4_layer.kernel)
        # print("layer4 shape", l4.shape)
        s1, s2 = tf.split(l4, [self.n_hidden1, self.n_hidden2], 1)
        self.l5_layer = tf.keras.layers.Dense(self.n_input1, kernel_initializer=self._init, name='layer5')
        l5 = self.l5_layer(s1)
        l5 = tf.keras.layers.BatchNormalization()(l5, training=bool(self.is_train))
        l5 = self.activation(l5)  # l1
        self.n_layer5 = l5.shape
        self.l5 = l5
        #self.trainable_variables.append(self.l5_layer.kernel)
        #self.trainable_variables.append(self.l5_layer.bias)
        #self.trainable_variables.append(self.l5_layer.kernel)
        # print("layer5 shape", l5.shape)
        self.l6_layer = tf.keras.layers.Dense(self.n_input2, kernel_initializer=self._init, name='layer6')
        l6 = self.l6_layer(s2)
        l6 = tf.keras.layers.BatchNormalization()(l6, training=bool(self.is_train))
        l6 = self.activation(l6)  # l2
        self.n_layer6 = l6.shape
        self.l6 = l6
        #self.trainable_variables.append(self.l6_layer.kernel)
        #self.trainable_variables.append(self.l6_layer.bias)
        #self.trainable_variables.append(self.l6_layer.kernel)
        # print("layer6 shape", l6.shape)
        return l5, l6

    #def modelCreate(self, X1, X2):
        #self.model_H = self.encode(X1, X2)
       # print("-------------------model_H  is type-----------------------------", type(self.model_H))
        #model = tf.keras.Model(inputs=[X1, X2], outputs=self.model_H)
       # return model

    def encodefun(self,X1,X2):

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

        # lasso
        lasso = self.L1regularization(self.W1) + self.L1regularization(self.W2) + \
                self.L1regularization(self.Wsh) + self.L1regularization(self.Wsht) + \
                self.L1regularization(self.W1t) + self.L1regularization(self.W2t)

        error = tf.losses.mean_squared_error(X1, X1_) + tf.losses.mean_squared_error(X2, X2_)
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

        # lasso
        lasso = self.L1regularization(self.W1) + self.L1regularization(self.W2) + \
                self.L1regularization(self.Wsh) + self.L1regularization(self.Wsht) + \
                self.L1regularization(self.W1t) + self.L1regularization(self.W2t)

        # reconstruction Error
        error = tf.losses.mean_squared_error(X1, X1_) + tf.losses.mean_squared_error(X2, X2_)

        # Loss function
        cost = 0.5 * error + 0.5 * self.lamda * (1 - self.alpha) * sgroup_lasso + 0.5 * self.lamda * self.alpha * lasso
        return cost

    def train(self):
       # save_sess = self.sess

        # costs history:
        costs = []
        costs_inter = []

        # for early stopping:
        best_cost = 100000
        stop = False
        last_improvement = 0

        n_samples = self.training_data1.shape[0]  # size of the training set
        # =============================================================================
        # train the mini_batches model using the early stopping criteria
        # =============================================================================
        epoch = 0
        counter = 0
        while epoch < self.max_epochs and stop == False:
            # train the model on the training set by mini batches
            # shuffle then split the training set to mini-batches of size self.batch_size
            seq = list(range(n_samples))
            random.shuffle(seq)
            mini_batches = [
                seq[k:k + self.batch_size]
                for k in range(0, n_samples, self.batch_size)
            ]

            avg_cost = 0.  # The average cost of mini_batches

            for sample in mini_batches:
                batch_xs1 = self.training_data1[sample][:]
                batch_xs2 = self.training_data2[sample][:]
                self.is_train=True
                # feed_dictio = {self.X1: batch_xs1, self.X2: batch_xs2, self.is_train: True}
                # cost = self.sess.run([self.loss_, self.train_step], feed_dict=feed_dictio)
                # avg_cost += cost[0] * len(sample) / n_samples

                with tf.GradientTape() as tape:
                    # 计算当前批次的损失
                    if epoch == 0 and counter == 0:
                       current_loss = self.loss(batch_xs1, batch_xs2)
                    else:
                       current_loss = self.lossfun(batch_xs1, batch_xs2)

                # 计算梯度
                #print("-------------------self.trainable_variables-------------------------")
                #print(self.trainable_variables)
                gradients = tape.gradient(current_loss, self.trainable_variables)
                #print("~~~~~~~~~~~~~~~~~~~~G value exists~~~~~~~~~~~~~~~~")
                #print(gradients)
                #print("~~~~~~~~~~~~~~~~~~~~G  value ends~~~~~~~~~~~~~~~~")
                # 更新模型的参数
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
               # cost = self.loss(batch_xs1, batch_xs2)
                cost = tf.reduce_mean(self.lossfun(batch_xs1, batch_xs2))
                self.batch_xs1 = batch_xs1
                self.batch_xs2 = batch_xs2
                avg_cost += cost * len(sample) / n_samples
                counter += 1
                # print("-----------------------Kosten---------------------------",cost)
                # print("---------------------------------------------")
                # avg_cost += cost[0] * len(sample) / n_samples

            # cost history since the last best cost
            costs_inter.append(avg_cost)

            # early stopping based on the validation set/ max_steps_without_decrease of the loss value: require_improvement
            if avg_cost < best_cost:
                #save_sess = self.sess  # save session
                best_cost = avg_cost
                costs += costs_inter  # costs history of the whole dataSet
                last_improvement = 0
                costs_inter = []
            else:
                last_improvement += 1

                #costs_val += costs_val_inter  # costs history of the validation set
                #costs += costs_inter
                #costs_val_inter = []
                #costs_inter = []
            if last_improvement > self.require_improvement:
                # print("No improvement found during the ( self.require_improvement) last iterations, stopping optimization.")
                # Break out from the loop.
                stop = True
                #self.sess = save_sess  # restore session with the best cost
            epoch += 1

        # =====================================End of model training ========================================
        #feed_dictio = {self.X1: self.training_data1, self.X2: self.training_data2, self.is_train: False}
       # costfinal, res = self.sess.run([self.loss_, self.H], feed_dict=feed_dictio)

        costfinal = tf.reduce_mean(self.lossfun(self.training_data1, self.training_data2))
        res = self.H

        plt.figure()
        plt.plot(costs)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.title("Learning rate =" + str(round(self.learning_rate, 3)))
        plt.savefig(r'C:\Users\gklizh\Documents\Workspace\code_and_data15\figure\loss_curve\getscores_picture' + str(self.mainer)+str(iterator) + '_test20.png')
        plt.close()

       # print("best_cost:", best_cost)
        #print("costfinal:", costfinal)
       # print("train done")
        return costfinal, res

    def Main(self, params):
        # retrieve parameters
        self.batch_size = params['batch_size']
        # print("self.batch_size",self.batch_size)#16
        self.n_hidden1 = params['units1']
        self.n_hidden2 = params['units2']
        self.alpha = params['alpha']
        self.lamda = params['lamda']
        self.learning_rate = params['learning_rate']

        # k fold validation
        k = 5
        self.require_improvement = 25
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

        k = 5
        print("H.layer1:", self.n_hidden1, ", H.layer2:", self.n_hidden2)
        print("k", k, "lamda", self.lamda, ", batch_size:", self.batch_size, ", alpha:", self.alpha, ", learning_rate:",
              self.learning_rate)
        print("initializer: ", init, ', optimizer:', opt)

        #tf.reset_default_graph()

        #self.X1 = tf.placeholder("float", shape=[None, self.training_data1.shape[1]])
        #self.X2 = tf.placeholder("float", shape=[None, self.training_data2.shape[1]])
        self.is_train = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool, name="is_train")

        #self.loss_ = self.loss(self.X1, self.X2)

       #if opt == 'Momentum':
            #self.train_step = self.optimizer(self.learning_rate, 0.9).minimize(self.loss_)
        #else:
            #self.train_step = self.optimizer(self.learning_rate).minimize(self.loss_)
        # Initiate a tensor session
        #init = tf.global_variables_initializer()
        #self.sess = tf.Session()
        #self.sess.run(init)

        # train the model

        loss, res = self.train()
        self.mainer +=1

        #e = shap.DeepExplainer(([self.X1, self.X2], self.H),
                               #[self.training_data1, self.training_data2],
                               #session=self.sess, learning_phase_flags=[self.is_train])

        #input_data1 = tf.keras.Input(shape=(self.training_data1.shape[1],))
        #input_data2 = tf.keras.Input(shape=(self.training_data2.shape[1],))
        #model = self.modelCreate(input_data1,input_data2)
        #e = shap.DeepExplainer(model, data=[self.training_data1, self.training_data2])
        #shap_values = e.shap_values([self.training_data1, self.training_data2])
        #self.sess.close()
        #tf.reset_default_graph()
        #del self.sess
        #return tloss, tres, shap_values
        return loss, res


if __name__ == '__main__':
    #selected_features = np.genfromtxt('./data/selected_features.csv', delimiter=',', skip_header=1)
    selected_features = np.genfromtxt(
        r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\data\selected_features.csv',
        delimiter=',',
        skip_header=1)
    # log10(htseq_fpkm + 1)
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

    namehtseq = np.genfromtxt(r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\data\exp_feature_names.csv', dtype=np.unicode_, delimiter=',', skip_header=1)
    namemethy = np.genfromtxt(r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\data\mty_feature_names.csv', dtype=np.unicode_, delimiter=',', skip_header=1)

    # tanh activation function
    act = tf.nn.tanh

    #fname = './result/32run/comm_trials_binary.pkl'
    #fname = (r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\result\comm_trials_binary_test.pkl')
    fname = (r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\result\comm_trials_binary_test19.pkl')
    #comm_trials_binary.pkl
    input = open(fname, 'rb')

    # run the multiModal autoEncoder
    for iterator in range(21):
        print("-------------------------------------------------------------------------------")
        print('iteration', iterator)
        cmtname = 'cmt' + str(iterator+1)
        selected_feat_cmt = selected_features[np.where(selected_features[:, 0] == iterator + 1)[0], :]

        print('first source ...')
        htseq_cmt = selected_feat_cmt[np.where(selected_feat_cmt[:, 1] == 1)[0], :]
        htseq_nbr = len(htseq_cmt)
        htseq_sel_data = inputhtseq[:, htseq_cmt[:, 2].astype(int) - 1]
        namehtseq_sel = namehtseq[htseq_cmt[:, 2].astype(int) - 1]

        print("second source ...")
        methy_cmt = selected_feat_cmt[np.where(selected_feat_cmt[:, 1] == 2)[0], :]
        methy_nbr = len(methy_cmt)
        methy_sel_data = inputmethy[:, methy_cmt[:, 2].astype(int) - 1]
        namemethy_sel = namemethy[methy_cmt[:, 2].astype(int) - 1]

        print("features size of the 1st dataset:", htseq_nbr)
        print("The data type of data1 is ",type(htseq_nbr))
        print("features size of the 2nd dataset:", methy_nbr)
        print("The data type of data2 is ", methy_nbr)

        n_hidden1 = htseq_nbr
        n_hidden2 = methy_nbr

        # if htseq_nbr > 1 and methy_nbr > 1 and cnv_seg_nbr > 1:
        if htseq_nbr > 1 and methy_nbr > 1:
            sae = AutoEncoder(data1 = htseq_sel_data, data2 = methy_sel_data, n_hiddensh = 1, activation=act)
            # train the HP optimization
            # get the loss of training the model on test data
            trials = pickle.load(input)
            best_loss = 1000
            best = trials.best_trial['result']['params']
           # loss, h, shapfeat = sae.Main(best)
            loss, h = sae.Main(best)
            #loss_scalar = loss[0]
            if loss < best_loss:
                best_loss = loss
                best_h = h
                #best_shapfeat = shapfeat

                if iterator == 0:
                    cmt_scores = best_h
                else:
                    cmt_scores = np.concatenate((cmt_scores, best_h), axis=1)

                #with open(r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\shap_doc\\' + cmtname + 'shapfeat_htseq.csv', 'w') as csvfile:
                   # writer = csv.writer(csvfile)
                   # [writer.writerow(r) for r in np.vstack((namehtseq_sel.reshape(1, len(namehtseq_sel)),
                                                          # shapfeat[0][0]))]

                #with open(r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\shap_doc\\' + cmtname + 'shapfeat_methy.csv', 'w') as csvfile:
                   # writer = csv.writer(csvfile)
                   # [writer.writerow(r) for r in np.vstack((namemethy_sel.reshape(1, len(namemethy_sel)),
                                                           #shapfeat[0][1]))]

                #htseqcont = np.mean(np.absolute(shapfeat[0][0]), axis=0)
                #reshtseq = np.hstack((namehtseq_sel.reshape(len(namehtseq_sel), 1),
                                      #htseqcont.reshape(len(htseqcont), 1)))
                #print('htseq shap done')

                #methycont = np.mean(np.absolute(shapfeat[0][1]), axis=0)
                #resmethy = np.hstack((namemethy_sel.reshape(len(namemethy_sel), 1),
                                               #methycont.reshape(len(methycont), 1)))
                #print('methy shap done')

                del htseq_sel_data
                del methy_sel_data
                del sae

                #fname = r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\shap_doc\cmt_SHAPValues.pkl'
                #pickle.dump(shapfeat, open(fname, "wb"))
                with open(r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\result\community\communityScores_compare19.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    [writer.writerow(r) for r in cmt_scores]
                #with open(r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\shap_doc\\' + cmtname + 'reshtseq.csv', 'w') as csvfile:
                   # writer = csv.writer(csvfile)
                    #[writer.writerow(r) for r in reshtseq]
                #with open(r'C:\Users\gklizh\Documents\Workspace\code_and_data15\data\python_related\shap_doc\\' + cmtname + 'resmethy.csv', 'w') as csvfile:
                    #writer = csv.writer(csvfile)
                   # [writer.writerow(r) for r in resmethy]