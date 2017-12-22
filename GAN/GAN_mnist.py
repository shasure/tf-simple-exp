#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author : zsy
Date : 2017/12/17"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import utils
import matplotlib.pyplot as plt

tf.flags.DEFINE_string('data_dir', '/datasets/mnist-data', 'mnist data')
tf.flags.DEFINE_string('train_dir', '/train_dir/gan-mnist', 'events and ckpt dir')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size m')
tf.flags.DEFINE_boolean('train', True, 'if training')
tf.flags.DEFINE_integer('num_steps', 30000, 'traing steps')

FLAGS = tf.flags.FLAGS


class GANmnist(object):
    def __init__(self):
        self.mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        self.netbuilder = utils.NetBuilder()
        self.z_dim = 100
        self.g_h1_units = 128
        self.d_h1_units = 128
        self.x_dim = self.mnist.train.images.shape[1]
        self.lr = 0.001
        self.k = 1

    def inference(self):
        self.z = tf.random_uniform([FLAGS.batch_size, self.z_dim], -1., 1.)
        self.x = tf.placeholder(tf.float32, [None, self.x_dim])
        # z
        with tf.variable_scope('generator'):
            g_h1 = self.netbuilder.affine(self.z, self.z_dim, self.g_h1_units, tf.nn.relu, 'g_h1')
            self.gen_input = self.netbuilder.affine(g_h1, self.g_h1_units, self.x_dim, tf.nn.sigmoid, 'gen')
            gen_images = tf.reshape(self.gen_input, shape=[-1, 28, 28, 1])
            tf.summary.image('gen_images', gen_images, max_outputs=10)
        with tf.variable_scope('discriminator') as scope:
            d_h1_z = self.netbuilder.affine(self.gen_input, self.x_dim, self.d_h1_units, tf.nn.relu, 'd_h1')
            z_logit = self.netbuilder.affine(d_h1_z, self.d_h1_units, 1, tf.nn.sigmoid, 'dis')
            # x
            scope.reuse_variables()
            d_h1_x = self.netbuilder.affine(self.x, self.x_dim, self.d_h1_units, tf.nn.relu, 'd_h1')
            x_logit = self.netbuilder.affine(d_h1_x, self.d_h1_units, 1, tf.nn.sigmoid, 'dis')
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

        opt_d = tf.train.AdamOptimizer()
        opt_g = tf.train.AdamOptimizer()
        param_d = [v for v in tf.trainable_variables() if not v.name.startswith('generator')]
        print('param_d', [v.name for v in param_d])
        grads_d = tf.gradients(d_loss, param_d)
        param_g = [v for v in tf.trainable_variables() if not v.name.startswith('discriminator')]
        print('param_g', [v.name for v in param_g])
        grads_g = tf.gradients(g_loss, param_g)
        train_op_d = opt_d.apply_gradients(zip(grads_d, param_d), global_step=global_step)
        with tf.Graph.control_dependencies(tf.get_default_graph(), [train_op_d]):
            train_op_g = opt_g.apply_gradients(zip(grads_g, param_g))

        with tf.Session() as sess:
            i = 0
            utils.init_or_load_var(None, sess, None, global_step, initializer=tf.random_normal_initializer(stddev=0.02))
            for local_step in range(FLAGS.num_steps):
                batch_x, _ = self.mnist.train.next_batch(FLAGS.batch_size)
                feed_dict = {self.x: batch_x}
                _, _, d_loss_str, g_loss_str, global_step_str = sess.run(
                    [train_op_d, train_op_g, d_loss, g_loss, global_step], feed_dict=feed_dict)
                if global_step_str % 100 == 0:
                    print('step: %d\t\td_loss: %.2f\t\tg_loss: %.2f' % (global_step_str, d_loss_str, g_loss_str))
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, global_step_str)
                if global_step_str % 1000 == 0:
                    self.generate(sess, feed_dict, i)
                    i += 1

    def generate(self, sess, feed_dict, index):
        samples = sess.run(self.gen_input, feed_dict=feed_dict)

        fig = utils.plot(samples[0:16, :])
        path = os.path.join(FLAGS.train_dir, 'out/')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path + '{}.png'.format(str(index).zfill(3))), bbox_inches='tight')
        plt.close(fig)


def main(_):
    gan = GANmnist()
    gan.train()


if __name__ == '__main__':
    tf.app.run()
