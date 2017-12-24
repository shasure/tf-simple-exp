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

import utils

tf.flags.DEFINE_string('data_dir', '/media/data/zsy/datasets/img_align_celeba', 'celeba data')
tf.flags.DEFINE_string('train_dir', '/home/zsy/train_dir/dcgan-celeba', 'events and ckpt dir')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size m')
tf.flags.DEFINE_boolean('train', True, 'if training')
tf.flags.DEFINE_integer('num_steps', 30000, 'traing steps')

FLAGS = tf.flags.FLAGS


class DCGANceleba(object):
    def __init__(self):
        self.celeba = self.getdataset()
        self.netbuilder = utils.NetBuilder()
        self.z_dim = 100
        self.g_size = [4, 8, 16, 32, 64]
        self.g_channels = [1024, 512, 256, 128, 3]
        self.g_last_channel = 1
        self.d_channels = [64, 128, 256, 512]
        self.d_last_channel = 1
        self.x_dim = 64 * 64 * 3  # cropped celeba image size
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
        g_out4 = tf.nn.relu(tf.layers.batch_normalization(g_out4, training=training))

        g_out5 = self.netbuilder.conv2d_trans(g_out4, [5, 5], [2, 2],
                                              [tf.shape(input)[0], self.g_size[4], self.g_size[4], self.g_channels[4]])

        output = tf.nn.tanh(g_out5)
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
        # self.x = tf.placeholder(tf.float32, [None, self.x_dim])
        # z
        with tf.variable_scope('generator'):
            self.gen_input = self.generator(self.z)
            gen_images = tf.reshape(self.gen_input, shape=[-1, 64, 64, 3])
            tf.summary.image('gen_images', gen_images, max_outputs=10)
        with tf.variable_scope('discriminator'):
            z_logit = self.discriminator(self.gen_input)
            # x
        with tf.variable_scope('discriminator', reuse=True):
            x = tf.reshape(self.x, [-1, 64, 64, 3])
            tf.summary.image('croped input images', x, max_outputs=10)
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

        training_iterator = self.celeba.make_one_shot_iterator()
        self.x = training_iterator.get_next()

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
            utils.init_or_load_var(None, sess, None, global_step, initializer=tf.random_normal_initializer(stddev=0.02))

            for local_step in range(FLAGS.num_steps):
                # # D
                # _, d_loss_str, global_step_str = sess.run(
                #     [train_op_d, d_loss, global_step])
                #
                #
                # # update G twice
                # for i in range(self.k):
                #     _, g_loss_str = sess.run(
                #         [train_op_g, g_loss])
                _, d_loss_str, global_step_str, _ ,g_loss_str = sess.run(
                    [train_op_d, d_loss, global_step, train_op_g, g_loss])


                if global_step_str % 100 == 0:
                    print('step: %d\t\td_loss: %.2f\t\tg_loss: %.2f' % (global_step_str, d_loss_str, g_loss_str))
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, global_step_str)
                # if global_step_str % 1000 == 0:
                #     self.generate(sess, index)
                #     index += 1

    def generate(self, sess, index):
        samples = sess.run(self.gen_input)

        fig = utils.plot(samples[0:16, :])
        path = os.path.join(FLAGS.train_dir, 'out/')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path + '{}.png'.format(str(index).zfill(3))), bbox_inches='tight')
        plt.close(fig)

    def getdataset(self):
        def _crop_resize_function(filename):
            # filename = filename[0]
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_image(image_string)
            image_decoded.set_shape([178, 218, 3])
            # image_cropped = tf.image.resize_image_with_crop_or_pad(image_decoded, 108, 108)
            image_resized = tf.image.resize_images(image_decoded, [64, 64])
            image_flatten = tf.reshape(image_resized, [64 * 64 * 3])
            image_casted = tf.cast(image_flatten, tf.float32)
            image_rescaled = image_casted / 127.5 - 1.
            return image_rescaled

        fpattern = '*.jpg'
        filenames = tf.constant(tf.gfile.Glob(os.path.join(FLAGS.data_dir, fpattern)))
        print('filenames shape ', filenames)
        dataset = tf.data.Dataset.from_tensor_slices(filenames, )
        print(dataset.output_shapes)
        print(dataset.output_types)
        dataset = dataset.map(_crop_resize_function)
        print(dataset.output_shapes)
        print(dataset.output_types)
        dataset = dataset.shuffle(buffer_size=10000).repeat(None).batch(FLAGS.batch_size)
        return dataset


def main(_):
    gan = DCGANceleba()
    gan.train()


if __name__ == '__main__':
    tf.app.run()
