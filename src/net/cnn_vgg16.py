
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def speech_cnn_arg_scope(is_training, weight_decay=0.0005):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    # Add normalizer_fn=slim.batch_norm if Batch Normalization is required!
    with slim.arg_scope([slim.conv3d, slim.fully_connected],
                        activation_fn=None,
                        normalizer_fn=slim.batch_norm,
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG'),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv3d], padding='VALID') as arg_sc:
            return arg_sc


def PReLU(input, scope):
    """
    Similar to TFlearn implementation
    :param input: input of the PReLU which is output of a layer.
    :return: The output.
    """
    alphas = tf.get_variable(scope, input.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)

    return tf.nn.relu(input) + alphas * (input - abs(input)) * 0.5


def speech_cnn(inputs, num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='cnn'):
    """Oxford Net VGG 11-Layers version A Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    """
    print('inputs1=', inputs)
    end_points = {}
    with tf.variable_scope(scope, 'net', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'

        # Collect outputs for conv2d and max_pool2d.
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.max_pool2d],  ####max_pool2d
                                            outputs_collections=end_points_collection):
            ##### Convolution Section #####
            inputs = tf.to_float(inputs)

            ############ Conv-1 ###############
            net = slim.conv3d(inputs, 64, [1, 3, 3], stride=[1, 1, 1], scope='conv11')
            net = PReLU(net, 'conv11_activation')
            print("1.1:", net)

            net = slim.conv3d(net, 64, [1, 3, 3], stride=[1, 1, 1], scope='conv12')
            net = PReLU(net, 'conv12_activation')
            net = tf.nn.max_pool3d(net, strides=[1, 1, 2, 2, 1], ksize=[1, 1, 2, 2, 1], padding='SAME', name='pool1')
            print('net 1st conv3d=', net)

            ############ Conv-2 ###############
            net = slim.conv3d(net, 128, [3, 3, 3], stride=[1, 1, 1], scope='conv21')
            net = PReLU(net, 'conv21_activation')
            print("2.1:", net)

            net = slim.conv3d(net, 128, [1, 3, 3], stride=[1, 1, 1], scope='conv22')
            net = PReLU(net, 'conv22_activation')
            net = tf.nn.max_pool3d(net, strides=[1, 1, 2, 2, 1], ksize=[1, 1, 2, 2, 1], padding='SAME', name='pool2')
            print('net 2nd conv3d=', net)

            ############ Conv-3 ###############
            net = slim.conv3d(net, 256, [1, 3, 3], stride=[1, 1, 1], scope='conv31')
            net = PReLU(net, 'conv31_activation')
            print("3.1:", net)

            net = slim.conv3d(net, 256, [1, 3, 3], stride=[1, 1, 1], scope='conv32')
            net = PReLU(net, 'conv32_activation')
            print("3.2:", net)

            net = slim.conv3d(net, 256, [1, 1, 1], stride=[1, 1, 1], scope='conv33')
            net = PReLU(net, 'conv33_activation')
            net = tf.nn.max_pool3d(net, strides=[1, 1, 2, 2, 1], ksize=[1, 1, 2, 2, 1], padding='SAME', name='pool3')
            # net = slim.max_pool2d(net, [1, 1], stride=[4, 1], scope='pool1')
            print('net 3rd conv3d=', net)
            ############ Conv-4 ###############
            net = slim.conv3d(net, 512, [1, 3, 3], stride=[1, 1, 1], scope='conv41')
            net = PReLU(net, 'conv41_activation')
            print("4.1:", net)

            net = slim.conv3d(net, 512, [3, 3, 3], stride=[1, 1, 1], scope='conv42')
            net = PReLU(net, 'conv42_activation')
            print("4.2:", net)

            net = slim.conv3d(net, 512, [3, 3, 3], stride=[1, 1, 1], scope='conv43')
            net = PReLU(net, 'conv43_activation')

            net = tf.nn.max_pool3d(net, strides=[1, 1, 2, 2, 1], ksize=[1, 1, 2, 2, 1], padding='SAME', name='pool4')
            # net = slim.max_pool2d(net, [1, 1], stride=[4, 1], scope='pool1')
            print('net 4th conv3d=', net)
            ############ Conv-5 ###############
            net = slim.conv3d(net, 512, [3, 3, 3], stride=[1, 1, 1], scope='conv51')
            net = PReLU(net, 'conv51_activation')
            print("5.1:", net)
            net = slim.conv3d(net, 512, [1, 3, 3], stride=[1, 1, 1], scope='conv52')
            net = PReLU(net, 'conv52_activation')
            print("5.2:", net)
            net = slim.conv3d(net, 512, [1, 1, 1], stride=[1, 1, 1], scope='conv53')
            net = PReLU(net, 'conv53_activation')

            net = tf.nn.max_pool3d(net, strides=[1, 1, 2, 2, 1], ksize=[1, 1, 2, 2, 1], padding='SAME', name='pool5')
            print('net 5th conv3d=', net)

            net = tf.contrib.layers.conv3d(net, 4096, [1, 3, 3], padding='VALID', activation_fn=None,scope='fc1')  ##############
            net = PReLU(net, 'fc1_activation')
            print('full1=', net)
            net = tf.contrib.layers.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout1')

            net = tf.contrib.layers.conv3d(net, 4096, [1, 1, 1], activation_fn=None, scope='fc2')
            net = PReLU(net, 'fc2_activation')
            net = tf.contrib.layers.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout2')
            print('full lost=', net)
            logits = tf.contrib.layers.conv3d(net, num_classes, [1, 1, 1], activation_fn=None, normalizer_fn=None,scope='fc3')

            # logits = tf.contrib.layers.conv2d(net, num_classes, [7, 7],padding='VALID', activation_fn=None, scope='fc')##############

            # Return the collections as a dictionary
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            # Squeeze spatially to eliminate extra dimensions.(embedding layer)
            # print('logits before squeeze=',logits)
            if spatial_squeeze:
                logits = tf.squeeze(logits, [1, 2, 3], name='fc/squeezed')
                end_points[sc.name + '/fc'] = logits
            # print('logits after squeeze=',logits)

            return logits, end_points
