# -*- coding:utf-8 -*-
# 
# @Author: ijnmklpo
# @Time: 2018/8/28 下午10:27
# @Desc: 通过图片识别属性的模型，用lenet试一下


import os
from PIL import Image
from tensorflow.contrib.data import Iterator
import random
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import re

import global_configs
import image_bleach


class LeNetModel(object):
    class_num = None

    def __init__(self, class_num):
        self.class_num = class_num


    def cnn_stream(self, img_batch, keep_prob=0.5):
        regularizer = slim.l2_regularizer(0.1)

        with tf.name_scope('conv1'):
            net = slim.conv2d(img_batch, 16, [5, 5], weights_regularizer=regularizer)
            net = slim.max_pool2d(net, [2, 2])      # 尺寸 / 2
            # 出来一个（hight / 2 = 64 /2 = 32, width / 2 = 64 / 2 = 32, 16）的特征组
        with tf.name_scope('conv2'):
            net = slim.conv2d(net, 32, [3, 3], weights_regularizer=regularizer)
            net = slim.max_pool2d(net, [2, 2])      # 尺寸 / 2
            # 出来一个（hight / 2 = 32 / 2 = 16, width / 2 = 32 / 2 = 16, 32）的特征组
        with tf.name_scope('conv3'):
            net = slim.conv2d(net, 128, [3, 3], weights_regularizer=regularizer)
            net = slim.conv2d(net, 128, [3, 3], weights_regularizer=regularizer)
            net = slim.max_pool2d(net, [2, 1], stride=[2, 1])   # 高度 / 2
            # 出来一个（hight / 2 = 16 / 2 = 8, width =16, 128）的特征组
        with tf.name_scope('conv4'):
            net = slim.conv2d(net, 64, [3, 3], normalizer_fn=slim.batch_norm, weights_regularizer=regularizer)
            net = slim.conv2d(net, 64, [3, 3], normalizer_fn=slim.batch_norm, weights_regularizer=regularizer)
            net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool4')
            # 出来一个（hight = 8, width / 2 = 16 / 2 = 8, 64）的特征组
        with tf.name_scope('fc1'):
            net = slim.flatten(net)     # 展开成4096 * 3 = 12288
            net = slim.fully_connected(net, 1024, weights_regularizer=regularizer)
            net = slim.dropout(net, keep_prob)
        with tf.name_scope('fc2'):
            logits = slim.fully_connected(net, self.class_num)
        return logits


    def loss_define(self, logits, labels_onehot):
        with tf.name_scope('loss'):
            # tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=labels_onehot)

            # 如果有get_total_loss，不用返回也没问题，get_total_loss会将全局各个地方定义好的loss都相加。
            tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=labels_onehot)

            # -------------- 加个正则

            # keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # for key in keys:
            #     print(key.name)
            # print("--------------------")
            # --------------

            loss = tf.losses.get_total_loss()
            # tf.summary.scalar('loss', loss)

        return loss



    def get_optimizer(self, loss, learning_rate=0.00001):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        return optimizer


    # def accuracy(self, logits, labels, begin, size):
    #     size[1] = self.class_num
    #     p = tf.slice(logits, begin, size)
    #
    #     max_idx_p = tf.argmax(p, 1)
    #     max_idx_p = tf.cast(max_idx_p, dtype=tf.int32)
    #     correct_pred = tf.equal(max_idx_p, labels)
    #     acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #     return acc
    #
    #
    # def prediction(self, logits, begin, size):
    #     size[1] = self.class_num
    #     p = tf.slice(logits, begin, size)
    #     max_idx_p = tf.argmax(p, 1)
    #     max_idx_p = tf.cast(max_idx_p, dtype=tf.int32)
    #     return max_idx_p


if __name__ == '__main__':
    batch_size = 5
    attr_label_num = 30
    img_loader = image_bleach.ImageLoader([global_configs.pic2attr_tfrecord_train_path], batch_size, 100, 40000)
    train_dataset = img_loader.launch_tfrecord_dataset()
    train_iterator = train_dataset.make_one_shot_iterator()

    # ========================= 数据导入 =========================

    # =================== 用handle导入，feedble ===================
    # 构造一个可导入(feedble)的句柄占位符，可以通过这个将训练集的句柄或者验证集的句柄传入
    handle = tf.placeholder(tf.string, shape=[])
    iterator = Iterator.from_string_handle(handle, train_dataset.output_types,
                                           train_dataset.output_shapes)
    pic_name_batch, pic_class_batch, attr_label_batch, img_batch = iterator.get_next()
    # 从迭代器中出来的是一个二维数组，而用到的id、effect_len和label是要一个一维数组，需要reshape以下
    pic_name_batch = tf.reshape(pic_name_batch, [batch_size])
    pic_class_batch = tf.reshape(pic_class_batch, [batch_size])
    attr_label_batch = tf.reshape(attr_label_batch, [batch_size, attr_label_num])
    # ==================/ 用handle导入，feedble /==================


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.global_variables_initializer())  # 所有变量初始化
        # 获得训练集和验证集的引用句柄，后面导入数据到模型用
        [train_iterator_handle] = sess.run([train_iterator.string_handle()])       # , valid_iterator.string_handle(), test_iterator.string_handle()

        pic_name_batch_out, pic_class_batch_out, attr_label_batch_out, img_batch_out = sess.run(
            [pic_name_batch, pic_class_batch, attr_label_batch, img_batch], feed_dict={handle: train_iterator_handle})

        print(pic_name_batch_out)
        print(pic_class_batch_out)
        print(attr_label_batch_out)
        print(img_batch_out)
        print(img_batch_out.shape)
        my_img = Image.fromarray(img_batch_out[1])
        my_img.show()

        coord.request_stop()
        coord.join(threads)