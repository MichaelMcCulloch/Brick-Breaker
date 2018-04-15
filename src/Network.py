import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Q_Learner:
    def __init__(self, image_H, image_W, h_size, n_layer, k_dims, s_length, n_outputs, n_feat, n_act, scope, learning_rate = 0.001):

        tf.reset_default_graph()
        self.image_H = image_H
        self.image_W = image_W
        self.h_size = h_size
        self.n_feat = n_feat
        self.n_act = n_act
        self.scope = scope
        self.learning_rate = learning_rate

        self.h_size     = h_size
        self.n_layer    = n_layer
        self.k_dims     = [[a,a] for a in k_dims]
        self.s_length   = [[a,a] for a in s_length]
        self.n_outputs  = n_outputs
        

        self.batch_size     = tf.placeholder(tf.int32, shape = [], name='batch_size')
        self.trace_length   = tf.placeholder(tf.int32, name='trace_length')
        self.dropout_p      = tf.placeholder(tf.float32, name='dropout_p')
        self.images         = tf.placeholder(tf.float32, shape=[None, None, self.image_H, self.image_W, 3], name='images')
        self.all_images     = tf.reshape(self.images, [self.batch_size * self.trace_length, self.image_H, self.image_W, 3])

        self._make_convolutional_layers()
        self._init_game_feat_out()
        self._init_LSTM()
        self._define_loss()

    def _make_convolutional_layers(self):
        self.conv_layers = list()
        input_layer = slim.conv2d(
            self.all_images, 
            num_outputs=self.n_outputs[0],
            kernel_size=self.k_dims[0], 
            stride=self.s_length[0],
            padding='VALID', scope = self.scope+'_conv0'
        )
        self.conv_layers.append(input_layer)
        for i in range(1, self.n_layer):
            next_layer = slim.conv2d(
                self.conv_layers[i-1], num_outputs = self.n_outputs[i],
                kernel_size=self.k_dims[i], stride=self.s_length[i],
                padding='VALID', scope = self.scope+'_conv' + str(i)
            )
            self.conv_layers.append(next_layer)

    def _init_game_feat_out(self):
        self.layer_conv_to_feat = tf.nn.dropout(
            slim.fully_connected(
                slim.flatten(self.conv_layers[-1]),
                512,
                scope=self.scope+'_conv_to_feat'
            ),
            self.dropout_p
        )
        self.flat_game_features = slim.fully_connected(
            self.layer_conv_to_feat,
            self.n_feat,
            scope= self.scope + '_flat_features',
            activation_fn = None
        )
        self.game_features_out = tf.reshape(
            self.flat_game_features,
            shape = [self.batch_size, self.trace_length, self.n_feat]
        )

        self.game_features_in = tf.placeholder(
            tf.float32,
            name="game_features_in",
            shape=[None, None, self.n_feat]
        )

        delta = self.game_features_out - self.game_features_in

        self.feature_loss = tf.reduce_mean(tf.square(delta))

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
    
    def _define_loss(self):
        pass