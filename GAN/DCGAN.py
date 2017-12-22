#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author : zsy
Date : 2017/12/21"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import utils

tf.flags.DEFINE_string('data_dir', '/home/zsy/datasets/mnist-data', 'mnist data')
tf.flags.DEFINE_string('train_dir', '/home/zsy/train_dir/dcgan-mnist', 'events and ckpt dir')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size m')
tf.flags.DEFINE_boolean('train', True, 'if training')
tf.flags.DEFINE_integer('num_steps', 30000, 'traing steps')

FLAGS = tf.flags.FLAGS


class DCGANmnist(object):
    def __init__(self):
        self.mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        self.netbuilder = utils.NetBuilder()
        self.z_dim = 100
        self.g_size = [4, 7, 14, 28]
        self.g_channels = [512, 256, 128, 1]
        self.g_last_channel = 1
        self.d_channels = [64, 128, 256, 512]
        self.d_last_channel = 1
        self.x_dim = self.mnist.train.images.shape[1]
        self.lr = 0.0002
        # 2 G after 1 D
        self.k = 2

    def generator(self, input, training=True):
        g_out1 = self.netbuilder.affine(input, self.z_dim, self.g_channels[0] * self.g_size[0] * self.g_size[0],
                                        activation=tf.nn.relu)
        g_out1 = tf.reshape(g_out1, shape=[-1, self.g_size[0], self.g_size[0], self.g_channels[0]])

        g_out2 = self.netbuilder.conv2d_trans(g_out1, [5, 5], [2, 2],
                                              [tf.shape(input)[0], self.g_size[1], self.g_size[1], self.g_channels[1]])
        g_out2 = tf.nn.relu(tf.layers.batch_normalization(g_out2, training=training))

        g_out3 = self.netbuilder.conv2d_trans(g_out2, [5, 5], [2, 2],
                                              [tf.shape(input)[0], self.g_size[2], self.g_size[2], self.g_channels[2]])
        g_out3 = tf.nn.relu(tf.layers.batch_normalization(g_out3, training=training))

        g_out4 = self.netbuilder.conv2d_trans(g_out3, [5, 5], [2, 2],
                                              [tf.shape(input)[0], self.g_size[3], self.g_size[3], self.g_channels[3]])
        output = tf.nn.sigmoid(g_out4)
        return output

    def discriminator(self, input, training=True):
        d_out1 = tf.layers.conv2d(input, self.d_channels[0], [5, 5], strides=(2, 2), padding='same')
        d_out1 = tf.nn.leaky_relu(d_out1, alpha=0.2)

        d_out2 = tf.layers.conv2d(d_out1, self.d_channels[1], [5, 5], strides=(2, 2), padding='same')
        d_out2 = tf.nn.leaky_relu(tf.layers.batch_normalization(d_out2, training=training), alpha=0.2)

        d_out3 = tf.layers.conv2d(d_out2, self.d_channels[2], [5, 5], strides=(2, 2), padding='same')
        d_out3 = tf.nn.leaky_relu(tf.layers.batch_normalization(d_out3, training=training), alpha=0.2)

        d_out4 = tf.layers.conv2d(d_out3, self.d_channels[3], [5, 5], strides=(2, 2), padding='same')
        d_out4 = tf.nn.leaky_relu(tf.layers.batch_normalization(d_out4, training=training), alpha=0.2)

        d_out4 = tf.layers.flatten(d_out4)
        output = tf.layers.dense(d_out4, 1, activation=tf.nn.sigmoid)
        return output

    def inference(self):
        self.z = tf.random_uniform([FLAGS.batch_size, self.z_dim], -1., 1.)
        self.x = tf.placeholder(tf.float32, [None, self.x_dim])
        # z
        with tf.variable_scope('generator'):
            self.gen_input = self.generator(self.z)
            gen_images = tf.reshape(self.gen_input, shape=[-1, 28, 28, 1])
            tf.summary.image('gen_images', gen_images, max_outputs=10)
        with tf.variable_scope('discriminator'):
            z_logit = self.discriminator(self.gen_input)
            # x
        with tf.variable_scope('discriminator', reuse=True):
            x = tf.reshape(self.x, [-1, 28, 28, 1])
            x_logit = self.discriminator(x)
            tf.summary.scalar('x_logit', tf.reduce_mean(x_logit))
            tf.summary.scalar('z_logit', tf.reduce_mean(z_logit))
        return z_logit, x_logit

    def loss(self):
        z_logit, x_logit = self.inference()
        d_loss = tf.reduce_mean(-(tf.log(1e-10 + x_logit) + tf.log(1e-10 + 1 - z_logit)))
        # d_loss = tf.reduce_mean(-(tf.log(1e-10 + x_logit) -tf.log(1e-10 + z_logit)))
        # g_loss = tf.reduce_mean(tf.log(1e-10 + 1 - z_logit))
        g_loss = tf.reduce_mean(-tf.log(1e-10 + z_logit))
        return d_loss, g_loss

    def train(self):
        global_step = tf.train.get_or_create_global_step()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
        d_loss, g_loss = self.loss()
        tf.summary.scalar('d_loss', d_loss)
        tf.summary.scalar('g_loss', g_loss)
        # merge_all should behind all summaries
        summary_op = tf.summary.merge_all()

        opt_d = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        opt_g = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        param_d = [v for v in tf.trainable_variables() if not v.name.startswith('generator')]
        print('param_d', [v.name for v in param_d])
        grads_d = tf.gradients(d_loss, param_d)
        param_g = [v for v in tf.trainable_variables() if not v.name.startswith('discriminator')]
        print('param_g', [v.name for v in param_g])
        grads_g = tf.gradients(g_loss, param_g)
        train_op_d = opt_d.apply_gradients(zip(grads_d, param_d), global_step=global_step)
        train_op_g = opt_g.apply_gradients(zip(grads_g, param_g))

        with tf.Session() as sess:
            index = 0
            utils.init_or_load_var(None, sess, None, global_step)
            for local_step in range(FLAGS.num_steps):
                batch_x, _ = self.mnist.train.next_batch(FLAGS.batch_size)
                feed_dict = {self.x: batch_x}
                # D
                _, d_loss_str, global_step_str = sess.run(
                    [train_op_d, d_loss, global_step], feed_dict=feed_dict)

                # update G twice
                for i in range(self.k):
                    _, g_loss_str = sess.run(
                        [train_op_g, g_loss], feed_dict=feed_dict)

                if global_step_str % 100 == 0:
                    print('step: %d\t\td_loss: %.2f\t\tg_loss: %.2f' % (global_step_str, d_loss_str, g_loss_str))
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, global_step_str)
                if global_step_str % 1000 == 0:
                    self.generate(sess, feed_dict, index)
                    index += 1

    def generate(self, sess, feed_dict, index):
        samples = sess.run(self.gen_input, feed_dict=feed_dict)

        fig = utils.plot(samples[0:16, :])
        path = os.path.join(FLAGS.train_dir, 'out/')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path + '{}.png'.format(str(index).zfill(3))), bbox_inches='tight')
        plt.close(fig)


def main(_):
    gan = DCGANmnist()
    gan.train()


if __name__ == '__main__':
    tf.app.run()
