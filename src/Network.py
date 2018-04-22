import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Q_Learner:
    def __init__(self, image_H, image_W, h_size, n_layer, k_dims, s_length, n_outputs, n_act, scope, learning_rate=0.001):

        
        self.image_H = image_H
        self.image_W = image_W
        self.h_size = h_size
        self.n_act = n_act
        self.scope = scope
        self.learning_rate = learning_rate

        self.h_size = h_size
        self.n_layer = n_layer
        self.k_dims = [[a, a] for a in k_dims]
        self.s_length = [[a, a] for a in s_length]
        self.n_outputs = n_outputs

        self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
        self.trace_length = tf.placeholder(tf.int32, name='trace_length')
        self.dropout_p = tf.placeholder(tf.float32, name='dropout_p')
        #self.scalarInput
        self.images = tf.placeholder(tf.float32, shape=[None, self.image_H * self.image_W], name='images')
        #self.imageIn
        self.all_images = tf.reshape(self.images, [-1, self.image_H, self.image_W, 1])

        self._make_convolutional_layers()
        self._init_LSTM()
        self._define_loss()

    def _make_convolutional_layers(self):
        self.conv_layers = list()
        input_layer = slim.conv2d(
            self.all_images,
            num_outputs=self.n_outputs[0],
            kernel_size=self.k_dims[0],
            stride=self.s_length[0],
            padding='VALID', scope=self.scope+'_conv0'
        )
        self.conv_layers.append(input_layer)
        for i in range(1, self.n_layer):
            next_layer = slim.conv2d(
                self.conv_layers[i-1], num_outputs=self.n_outputs[i],
                kernel_size=self.k_dims[i], stride=self.s_length[i],
                padding='VALID', scope=self.scope+'_conv' + str(i)
            )
            self.conv_layers.append(next_layer)

    def _init_LSTM(self):
        self.conv_flattened = slim.flatten(self.conv_layers[-1])
        self.flat_size = int(self.conv_flattened.shape[1])
        self.conv_flat = tf.nn.dropout(
            tf.reshape(
                self.conv_flattened,
                [self.batch_size, self.trace_length, self.flat_size]
            ),
            self.dropout_p
        )

        self.cell = tf.nn.rnn_cell.LSTMCell(self.h_size)
        self.rnn_state_in = self.cell.zero_state(self.batch_size, tf.float32)

        rnn_output, self.rnn_state_out = tf.nn.dynamic_rnn(
            self.cell,
            self.conv_flat,
            initial_state=self.rnn_state_in,
            dtype=tf.float32,
            scope=self.scope+'_rnn'
        )

        self.rnn_output = tf.reshape(rnn_output, [-1, self.h_size])

        self.stream_A, self.stream_V = tf.split(self.rnn_output, 2, 1)
        self.AW = tf.Variable(tf.random_normal([self.h_size//2,4]))
        self.VW = tf.Variable(tf.random_normal([self.h_size//2,1]))
        self.Advantage = tf.matmul(self.stream_A,self.AW)
        self.Value = tf.matmul(self.stream_V,self.VW)
        
        self.salience = tf.gradients(self.Advantage,self.all_images)
        Q = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))

        self.Q = tf.reshape(Q, [self.batch_size, self.trace_length, self.n_act])
        self.choice = tf.argmax(self.Q, 2)
        self.max_Q = tf.reduce_max(self.Q, 2)

    def _define_loss(self):
        self.gamma = tf.placeholder(tf.float32, name='gamma')
        self.target_q = tf.placeholder(tf.float32, name='target_q', shape=[None, None])
        self.rewards = tf.placeholder(tf.float32, name='rewards', shape=[None, None])
        self.actions = tf.placeholder(tf.uint8, name='actions', shape=[None, None])
        self.actions_one_hot = tf.one_hot(self.actions, 3, dtype=tf.float32)
        y = self.rewards + self.gamma * self.target_q
        Qas = tf.reduce_sum(tf.one_hot(tf.argmax(self.actions_one_hot, 2), self.n_act) * self.Q, 2)
        self.ignore_up_to = tf.placeholder(tf.int32, name='ignore_up_to')
        y = tf.slice(y, [0, self.ignore_up_to], [-1, -1])
        Qas = tf.slice(Qas, [0, self.ignore_up_to], [-1, -1])
        self.td_error = tf.square(y-Qas)
        self.td_error_sum = tf.reduce_sum(self.td_error, 1)
        self.q_loss = tf.reduce_mean(self.td_error)
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.q_loss)


