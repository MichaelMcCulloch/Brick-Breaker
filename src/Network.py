import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Q_Learner:
    def __init__(self, image_H, image_W, h_size, n_layer, k_dims, s_length, n_feat, n_act, scope, learning_rate = 0.001):
        self.image_H = image_H
        self.image_W = image_W
        self.h_size = h_size
        self.n_feat = n_feat
        self.n_act = n_act
        self.scope = scope
        self.learning_rate = learning_rate

        self.batch_size     = tf.placeholder(tf.int32, shape = [], name='batch_size')
        self.trace_length   = tf.placeholder(tf.int32, name='trace_length')
        self.dropout_p      = tf.placeholder(tf.float32, name='dropout_p')
        self.images         = tf.placeholder(tf.float32, shape=[None, None, self.image_H, self.image_W, 3], name='images')
        self.all_images = tf.reshape(self.images, [self.batch_size * self.trace_length, self.image_H, self.image_W, 3])

        self._make_convolutional_layers(n_layer, k_dims, s_length)
        exit()
        self._init_game_feat_out()
        self._init_LSTM()
        self._define_loss()

    def _make_convolutional_layers(self, n_layer, k_dims, s_length, n_outputs = [32, 64, 128, 256]):
        print(k_dims, s_length)
        self.layers = list()
        input_layer = slim.conv2d(
            self.all_images, num_outputs=n_outputs[0],
            kernel_size=[k_dims[0],k_dims[0]], stride=[s_length[0],s_length[0]],
            padding='VALID', scope = self.scope+'_conv0'
        )
        self.layers.append(input_layer)
        for i in range(1, n_layer):
            next_layer = slim.conv2d(
                self.layers[i-1], num_outputs = n_outputs[i],
                kernel_size=[k_dims[i],k_dims[i]], stride=[s_length[i],s_length[i]],
                padding='VALID', scope = self.scope+'_conv' + str(i)
            )
            self.layers.append(next_layer)
        for i in self.layers:
            print(i)