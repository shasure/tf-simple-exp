#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author : zsy
Date : 2017/12/17"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec


class NetBuilder(object):
    def affine(self, input_layer, num_input_units, num_output_units, activation=None, scope=None):
        """

        :param input_layer:
        :param num_input_units:
        :param num_output_units:
        :param activation: activation is a function
        :param scope: variable scope
        :return: output units of the full connection layer
        """
        with tf.variable_scope(scope, default_name='affine'):
            weight = tf.get_variable('weight', [num_input_units, num_output_units])
            bias = tf.get_variable('bias', [num_output_units])
            xw_plus_b = tf.nn.xw_plus_b(input_layer, weight, bias)
            if activation:
                return activation(xw_plus_b)
            return xw_plus_b

    def conv2d_trans(self, input, kernel_shape, strides_shape, output_shape, scope=None):
        # NHWC
        with tf.variable_scope(scope, default_name='conv2d_trans'):
            filters = tf.get_variable(name='trans_conv_filters', shape=kernel_shape + [output_shape[-1], input.shape[-1]])
            output = tf.nn.conv2d_transpose(input, filters, output_shape=output_shape, strides=[1, strides_shape[0], strides_shape[1], 1])
            return output


def xavier_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


def init_or_load_var(saver, sess, ckpt_dir, global_step_tensor, initializer=xavier_init):
    if saver:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if os.path.isabs(ckpt.model_checkpoint_path):
                # Restores from checkpoint with absolute path.
                model_checkpoint_path = ckpt.model_checkpoint_path
            else:
                # Restores from checkpoint with relative path.
                model_checkpoint_path = os.path.join(ckpt_dir, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/imagenet_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            if not global_step.isdigit():
                global_step = 0
            else:
                global_step = int(global_step)
            saver.restore(sess, model_checkpoint_path)
            print('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)
            # assign to global step
            sess.run(global_step_tensor.assign(global_step))
    else:
        init_op = tf.global_variables_initializer()
        if not initializer:
            tf.get_variable_scope().set_initializer(xavier_init)
        sess.run(init_op)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig
