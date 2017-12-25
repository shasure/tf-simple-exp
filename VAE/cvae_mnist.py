#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author : zsy
Date : 2017/12/13"""
import os
import matplotlib

matplotlib.use('Agg')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

tf.flags.DEFINE_string('data_dir', '/home/zsy/datasets/mnist-data', 'mnist data')
tf.flags.DEFINE_string('train_dir', '/home/zsy/train_dir/cvae-mnist', 'events and ckpt dir')
tf.flags.DEFINE_integer('batch_size', 64, 'M in VAE')
tf.flags.DEFINE_boolean('train', True, 'if training')
tf.flags.DEFINE_boolean('p_sampling', False, 'if sampling when reconstruct X')
tf.flags.DEFINE_integer('num_steps', 30000, 'traing steps')

FLAGS = tf.flags.FLAGS


def init_or_load_var(saver, sess, ckpt_dir, global_step_tensor):
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
        sess.run(init_op)


def xavier_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


class CVAEMnist(object):
    def __init__(self):
        # only use mnist.train，do not need labels
        self.mnist = self.read_data()
        self.input_dim = self.mnist.train.images.shape[1]
        self.y_dim = self.mnist.train.labels.shape[1]  # add label info
        self.batch_size = FLAGS.batch_size
        self.num_steps = FLAGS.num_steps
        self.e_hidden1 = 512
        self.latent_dim = 2
        self.d_hidden1 = 512
        self.lr = 0.001
        self.config = tf.ConfigProto(log_device_placement=False)
        self.config.gpu_options.allow_growth = False
        self.config.allow_soft_placement = True

    def read_data(self):
        return input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    def inference(self, istrain=True):
        self.inputx = tf.placeholder(tf.float32, [None, self.input_dim])
        self.inputy = tf.placeholder(tf.float32, [None, self.y_dim])
        self.randz = tf.placeholder(tf.float32, [None, self.latent_dim])
        # q(z|x,y)
        e_hidden_w1 = tf.Variable(xavier_init([self.input_dim + self.y_dim, self.e_hidden1]))
        e_hidden_b1 = tf.Variable(xavier_init([self.e_hidden1]))
        concat_xy = tf.concat([self.inputx, self.inputy], axis=1)
        e_h = tf.nn.tanh(tf.nn.xw_plus_b(concat_xy, e_hidden_w1, e_hidden_b1))

        e_hidden_w2 = tf.Variable(xavier_init([self.e_hidden1, self.latent_dim]))
        e_hidden_b2 = tf.Variable(xavier_init([self.latent_dim]))
        e_mean = tf.nn.xw_plus_b(e_h, e_hidden_w2, e_hidden_b2)

        e_hidden_w3 = tf.Variable(xavier_init([self.e_hidden1, self.latent_dim]))
        e_hidden_b3 = tf.Variable(xavier_init([self.latent_dim]))
        e_logvar = tf.nn.xw_plus_b(e_h, e_hidden_w3, e_hidden_b3)

        tf.summary.histogram('e_logvar', e_logvar)

        epsilon = tf.random_normal(tf.shape(e_logvar))

        z = tf.add(e_mean, tf.multiply(tf.exp(e_logvar / 2), epsilon))

        # p(x|z,y)
        d_hidden_w1 = tf.Variable(xavier_init([self.latent_dim + self.y_dim, self.d_hidden1]))
        d_hidden_b1 = tf.Variable(xavier_init([self.d_hidden1]))
        concat_zy = tf.concat([z, self.inputy], axis=1)
        d_h = tf.nn.tanh(tf.nn.xw_plus_b(concat_zy, d_hidden_w1, d_hidden_b1))

        if not FLAGS.p_sampling:
            d_hidden_w2 = tf.Variable(xavier_init([self.d_hidden1, self.input_dim]))
            d_hidden_b2 = tf.Variable(xavier_init([self.input_dim]))
            reconstruction = tf.nn.sigmoid(tf.nn.xw_plus_b(d_h, d_hidden_w2, d_hidden_b2))
        else:
            d_hidden_w2 = tf.Variable(xavier_init([self.d_hidden1, self.input_dim]))
            d_hidden_b2 = tf.Variable(xavier_init([self.input_dim]))
            d_mean = tf.nn.xw_plus_b(d_h, d_hidden_w2, d_hidden_b2)

            d_hidden_w3 = tf.Variable(xavier_init([self.d_hidden1, self.input_dim]), name='eps_w')
            d_hidden_b3 = tf.Variable(xavier_init([self.input_dim]), name='eps_b')
            d_logvar = tf.nn.xw_plus_b(d_h, d_hidden_w3, d_hidden_b3)

            tf.summary.histogram('d_logvar', d_logvar)

            epsilon_d = tf.random_normal(tf.shape(d_mean))
            # if forgot sigmoid activation, genetated images will be gray
            reconstruction = tf.nn.sigmoid(tf.add(d_mean, tf.multiply(tf.exp(d_logvar / 2), epsilon_d)))

        # summary image
        if istrain:
            reconstruction_image = tf.reshape(reconstruction, [-1, 28, 28, 1])
            tf.summary.image('reconstr_image', reconstruction_image, 10)

            kl_loss = -0.5 * tf.reduce_sum(1 + e_logvar - tf.square(e_mean) - tf.exp(e_logvar), 1)
            return reconstruction, kl_loss
        else:  # generate
            concat_randzy = tf.concat([self.randz, self.inputy], axis=1)
            d_h_gen = tf.nn.tanh(tf.nn.xw_plus_b(concat_randzy, d_hidden_w1, d_hidden_b1))
            reconstr_gen = tf.nn.sigmoid(tf.nn.xw_plus_b(d_h_gen, d_hidden_w2, d_hidden_b2))
            tf.summary.image('gen_image', reconstr_gen, 10)
            return reconstr_gen

    def loss(self):
        reconstruction, kl_loss = self.inference()
        # loss: either MSE or cross entropy
        # main_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=self.inputx),1)
        # main_loss = -tf.reduce_sum(self.inputx * tf.log(1e-10 + reconstruction) + (1 - self.inputx) * tf.log(
        #     1e-10 + 1 - reconstruction), 1)

        # mean square error not work well, maybe change params will work
        # main_loss = tf.reduce_sum(tf.square(self.inputx - reconstruction), 1)
        main_loss = tf.reduce_sum(tf.squared_difference(self.inputx, reconstruction), 1)
        loss = tf.reduce_mean(main_loss + kl_loss)
        # summary losses
        tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss))
        tf.summary.scalar('main_loss', tf.reduce_mean(main_loss))
        tf.summary.scalar('losses', loss)
        return loss

    def train(self):
        global_step = tf.train.get_or_create_global_step()
        loss_op = self.loss()
        opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        # param = [p for p in tf.trainable_variables() if not p.name.startswith('eps')]
        param = tf.trainable_variables()
        print([p.name for p in param])
        grads = tf.gradients(loss_op, param)
        train_op = opt.apply_gradients(zip(grads, param), global_step=global_step)

        tf.summary.scalar('grad_norm', tf.global_norm(grads))
        tf.summary.scalar('lr', self.lr)
        # summary
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
        summary_op = tf.summary.merge_all()
        # saver
        saver = tf.train.Saver()

        with tf.Session(config=self.config) as sess:
            init_or_load_var(saver, sess, FLAGS.train_dir, global_step)
            for step in range(self.num_steps):
                # feed dict
                batch_x, batch_y = self.mnist.train.next_batch(self.batch_size)
                # print(batch_x[0])
                feed_dict = {self.inputx: batch_x, self.inputy: batch_y}
                loss, _, g_step = sess.run([loss_op, train_op, global_step], feed_dict=feed_dict)
                if g_step % 100 == 0:
                    # loss
                    print('step: %d \t\t loss: %.2f' % (g_step, loss))
                    # summary
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, g_step)
                if g_step % 100 == 0:
                    # ckpt
                    if not tf.gfile.Exists(FLAGS.train_dir):
                        tf.gfile.MakeDirs(FLAGS.train_dir)
                    saver.save(sess, os.path.join(FLAGS.train_dir, 'model.ckpt'), g_step)

    def generate(self):
        dtime = datetime.now().strftime('%d_%m_%Y_%H_%M')
        fig_dir = os.path.join(FLAGS.train_dir, 'out')
        if not tf.gfile.Exists(fig_dir):
            tf.gfile.MakeDirs(fig_dir)
        gen_x = self.inference(istrain=False)
        n = 20
        x_axis = np.linspace(-3, 3, n)
        y_axis = np.linspace(-3, 3, n)

        saver = tf.train.Saver()
        global_step = tf.train.get_or_create_global_step()
        canvas = np.empty((28 * n, 28 * n))
        with tf.Session(config=self.config) as sess:
            init_or_load_var(saver, sess, FLAGS.train_dir, global_step)
            for y in range(10):
                for i, xi in enumerate(x_axis):
                    for j, yi in enumerate(y_axis):
                        randz = np.array([[xi, yi]] * 1)
                        yarr = np.zeros(shape=[1, 10])
                        yarr[:, y] = 1.
                        x = sess.run(gen_x, feed_dict={self.randz: randz, self.inputy: yarr})
                        canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x[0].reshape(28, 28)
                plt.figure(figsize=(8, 10))
                plt.axis('off')
                plt.imshow(canvas, origin='upper', cmap='gray')
                plt.savefig(os.path.join(fig_dir, 'cvae_digit%d_%s.png' % (y, dtime)))


def main(_):
    vae = CVAEMnist()
    if FLAGS.train:
        vae.train()  # training
    else:
        vae.generate()  # generate images


if __name__ == '__main__':
    tf.app.run()
