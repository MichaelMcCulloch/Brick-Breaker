#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim 

from memory import Replay_Buffer

class Q_Learner():
    def __init__(self, image_H, image_W, h_size, n_Feat, n_Act, scope, learning_rate = 0.001, use_game_features=False, recurrent=True):
        self.image_H = image_H  # Image Dimensions
        self.image_W = image_W
        self.h_size = h_size
        self.n_Feat = n_Feat  # Number of game features to learn
        self.n_Act = n_Act  # Size of action space
        self.scope = scope  # Name of network
        self.learning_rate = learning_rate
        self.use_game_features = use_game_features
        self.recurrent = recurrent

        self.batch_size = tf.placeholder(
            dtype=tf.int32, name='batch_size', shape=[])
        self.trace_length = tf.placeholder(dtype=tf.int32, name='trace_length')
        self.dropout_probability = tf.placeholder(
            dtype=tf.float32, name='dropout_p')
        self.images = tf.placeholder(dtype=tf.float32, name='images', shape=[
                                     None, None, image_H, image_W, 3])  # 3 color channels

        self.all_images = tf.reshape(
            self.images, [self.batch_size * self.trace_length, image_H, image_W, 3])


        self._init_conv_layers()
        self._init_game_feat_out()

        if recurrent:
            self._init_recurrent_module()
        else:
            self._init_feed_forward_module()

        self._define_loss()

    def _init_conv_layers(self):
        self.conv1 = slim.conv2d(
            self.all_images, num_outputs=32,
            kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            scope=self.scope+'_conv1'
        )
        self.conv2 = slim.conv2d(
            self.conv1, num_outputs=64,
            kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            scope=self.scope+'_conv2'
        )

    def _init_game_feat_out(self):
        self.layer4 = tf.nn.dropout(
            slim.fully_connected(
                slim.flatten(self.conv2),
                512,
                scope=self.scope+'_layer4'
            ),
            self.dropout_probability
        )
        self.flat_game_features = slim.fully_connected(
            self.layer4,
            self.n_Feat,
            scope=self.scope+'_flat_game_feat',
            activation_fn=None
        )
        self.game_features_out = tf.reshape(
            self.flat_game_features,
            shape=[self.batch_size, self.trace_length, self.n_Feat]
        )

        self.game_features_in = tf.placeholder(
            tf.float32,
            name='game_features_in',
            shape=[None, None, self.n_Feat]
        )

        delta = self.game_features_out - self.game_features_in

        # optimize on RMS
        self.feature_loss = tf.reduce_mean(tf.square(delta))

    def _init_recurrent_module(self):
        self.layer3 = tf.nn.dropout(
            tf.reshape(
                slim.flatten(self.conv2),
                # size of flattened conv2 layer
                [self.batch_size, self.trace_length, 4608]
            ),
            self.dropout_probability
        )

        self.cell = tf.nn.rnn_cell.LSTMCell(self.h_size)
        self.state_in = self.cell.zero_state(self.batch_size, tf.float32)

        rnn_output, self.state_out = tf.nn.dynamic_rnn(
            self.cell,
            self.layer3,
            initial_state=self.state_in,
            dtype=tf.float32,
            scope=self.scope+'_rnn'
        )

        self.rnn_output = tf.reshape(rnn_output, [-1, self.h_size])

        # DUELING
        self.stream_Adv, self.stream_Val = tf.split(self.rnn_output, 2, 1)
        self.adv_W = tf.Variable(
            tf.random_normal([self.h_size//2, self.n_Act]))
        self.val_W = tf.Variable(tf.random_normal([self.h_size//2, 1]))
        self.Advantage = tf.matmul(self.stream_Adv, self.adv_W)
        self.Value = tf.matmul(self.stream_Val, self.val_W)

        # Recombine Streams into the Q value
        Q = self.Value + \
            tf.subtract(self.Advantage, tf.reduce_mean(
                self.Advantage, axis=1, keepdims=True))

        self.Q = tf.reshape(
            Q, [self.batch_size, self.trace_length, self.n_Act])
        self.choice = tf.argmax(self.Q, 2)
        self.max_Q = tf.reduce_max(self.Q, 2)

    def _init_feed_forward_module(self):
        self.layer3 = tf.nn.dropout(
            tf.reshape(
                slim.flatten(self.conv2),
                # size of flattened conv2 layer
                [self.batch_size * self.trace_length, 4608]
            ),
            self.dropout_probability
        )

        self.layer3_5 = slim.fully_connected(
            self.layer3,
            self.h_size,
            scope=self.scope+'_layer3_5'
        )

        # DUELING
        self.stream_Adv, self.stream_Val = tf.split(self.layer3_5, 2, 1)
        self.adv_W = tf.Variable(
            tf.random_normal([self.h_size//2, self.n_Act]))
        self.val_W = tf.Variable(tf.random_normal([self.h_size//2, 1]))
        self.Advantage = tf.matmul(self.stream_Adv, self.adv_W)
        self.Value = tf.matmul(self.stream_Val, self.val_W)

        # Recombine Streams into the Q value
        Q = self.Value + \
            tf.subtract(self.Advantage, tf.reduce_mean(
                self.Advantage, axis=1, keepdims=True))

        self.Q = tf.reshape(
            Q, [self.batch_size, self.trace_length, self.n_Act])

        # TODO: 2 is the axis over which to pick the max. Check this remains correct
        self.choice = tf.argmax(self.Q, 2)
        self.max_Q = tf.reduce_max(self.Q, 2)

    def _define_loss(self):
        self.gamma = tf.placeholder(tf.float32, name='gamma')
        self.target_Q = tf.placeholder(
            tf.float32, name='target_Q', shape=[None, None])
        self.rewards = tf.placeholder(
            tf.float32, name='rewards', shape=[None, None])
        self.actions = tf.placeholder(tf.float32, name='actions', shape=[
                                      None, None, self.n_Act])

        y = self.rewards + self.gamma * self.target_Q

        # TODO: Verify This!
        # Q(a,s)
        Q_as = tf.reduce_sum(
            tf.one_hot(
                tf.argmax(self.actions, 2),
                self.n_Act
            ) * self.Q,
            2
        )

        self.ignore_until = tf.placeholder(tf.int32, name='ignore_until')
        y = tf.slice(y, [0, self.ignore_until], [-1, -1])
        Q_as = tf.slice(Q_as, [0, self.ignore_until], [-1, -1])
        self.Q_Loss = tf.reduce_mean(tf.square(y - Q_as))

        if self.use_game_features:
            self.loss = self.Q_Loss + self.feature_loss
        else:
            self.loss = self.Q_Loss

        self.train_step = tf.train.RMSPropOptimizer(
            self.learning_rate).minimize(self.loss)


    

