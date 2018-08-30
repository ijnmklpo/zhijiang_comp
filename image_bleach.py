# -*- coding:utf-8 -*-
# 
# @Author: ryh-dell
# @Time: 2018/8/28 19:32
# @Desc: 


import numpy as np
import cv2
import os
from PIL import Image
import random
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.data as cb_data
import logging
from io import BytesIO, StringIO
import re
import pandas as pd

import global_configs




class TFrecordMaker(object):
    image_path = None
    img_attr_df = None
    output_train_path = None
    output_valid_path = None
    train_rat = None

    def __init__(self, image_path, img_attr_df, output_path, train_rat=1.0, shuffle=True):
        self.image_path = image_path
        self.img_attr_df = img_attr_df
        self.output_train_path = output_path
        self.train_rat = train_rat

        if train_rat > 0:
            self.output_train_path = re.sub('\.', '_train.', output_path)
            self.output_valid_path = re.sub('\.', '_valid.', output_path)

        if shuffle is True:
            self.img_attr_df.sample(frac=1)


    def make_tfrecord(self, image_path, img_attr_df, output_path):
        print(len(img_attr_df))
        with tf.python_io.TFRecordWriter(output_path) as writer:
            for i, line in img_attr_df.iterrows():
                print(i)
                img = Image.open(os.path.join(image_path, line['pic_name']))
                # 构建一个假的二进制缓存作为磁盘空间，然后把图片保存进去，再取出来变成二进制文件
                tmp_f = BytesIO()
                img.save(tmp_f, 'JPEG')
                img_bytes = tmp_f.getvalue()

                attr_label = (line['attr_label']).strip('[]').split(', ')
                attr_label = [float(item) for item in attr_label]

                pic_name_feat = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[bytes(line['pic_name'], 'utf-8')]))
                pic_class_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(line['class'], 'utf-8')]))
                attr_label_feat = tf.train.Feature(float_list=tf.train.FloatList(value=attr_label))
                img_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'pic_name_feat': pic_name_feat,
                            'pic_class_feat': pic_class_feat,
                            'attr_label_feat': attr_label_feat,
                            'img_feat': img_feat,
                        }
                    )
                )
                writer.write(example.SerializeToString())  # 序列化为字符串


    def package_dataset(self):
        tot_data_len = len(self.img_attr_df)

        self.make_tfrecord(self.image_path, self.img_attr_df[:int(tot_data_len*self.train_rat)], self.output_train_path)
        if self.train_rat < 1:
            self.make_tfrecord(self.image_path, self.img_attr_df[int(tot_data_len * self.train_rat):], self.output_valid_path)


class ImageLoader(object):
    tfrecord_path_list = None
    batch_size = None
    repeat_eposh_num = None
    shffle_buffer_size = None

    def __init__(self, tfrecord_path_list, batch_size, repeat_epoch_num, shuffle_buffer_size):
        self.tfrecord_path_list = tfrecord_path_list
        self.batch_size = batch_size
        self.repeat_epoch_num = repeat_epoch_num
        self.shuffle_buffer_size = shuffle_buffer_size


    def launch_tfrecord_dataset(self):
        def __parse(example_proto):
            features = {
                'pic_name_feat': tf.FixedLenFeature([1], tf.string),
                'pic_class_feat': tf.FixedLenFeature([1], tf.string),
                'attr_label_feat': tf.VarLenFeature(tf.float32),
                'img_feat': tf.FixedLenFeature([], tf.string),
            }
            parsed_features = tf.parse_single_example(example_proto, features)

            pic_name = parsed_features['pic_name_feat']
            pic_class = parsed_features['pic_class_feat']
            attr_label = parsed_features['attr_label_feat']
            img = parsed_features['img_feat']

            pic_name = tf.cast(pic_name, dtype=tf.string)
            pic_class = tf.cast(pic_class, dtype=tf.string)
            attr_label = tf.sparse_tensor_to_dense(attr_label)

            # img = tf.decode_raw(img, out_type=tf.uint8)       # 不知道这个解析方法跟img.decode_jpeg有什么区别
            img = tf.image.decode_jpeg(img, channels=3)
            # img = tf.image.per_image_standardization(img)       # 做个图像的归一化
            # img = tf.random_crop(img, (224, 224, 3))            # 固定尺寸的时候才用

            return pic_name, pic_class, attr_label, img

        data_set = cb_data.TFRecordDataset(self.tfrecord_path_list)
        # 这个比较奇怪不知道为什么必须要加个括号。。必须要是个tuple？
        parsed_dataset = (data_set.map(__parse))
        if self.repeat_epoch_num is not None:
            # 指定重复的次数，队列会以这个epoch次数为长度
            parsed_dataset = parsed_dataset.repeat(self.repeat_epoch_num)
        if self.shuffle_buffer_size is not None:
            # 指定shuffle的范围，一般要选比整个数据集的长度大，才能整体打乱
            parsed_dataset = parsed_dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        # 由于有变长数组，所以必须要padd，否则用parsed_dataset.batch(batch_size)就可以了
        # 像这个例子里，如果parse函数的返回为多个变量，则padded_shapes需要是一个tuple，每个元素对应该变量在该批次pad到的上限，-1为pad到最长
        # parsed_dataset = parsed_dataset.padded_batch(self.batch_size, padded_shapes=([-1], [-1], [-1], [-1, -1, 3]))
        parsed_dataset = parsed_dataset.batch(self.batch_size)
        return parsed_dataset   # 包含4个字段：pic_name, pic_class, attr_label, img




if __name__ == '__main__':
    # img_attr_df = pd.read_csv(global_configs.train_pic_class_attr_path, sep='\t')
    # tf_maker = TFrecordMaker(os.path.join(global_configs.root_path, 'pics'), img_attr_df, global_configs.pic2attr_tfrecord_path, train_rat=0.8)
    #
    # tf_maker.package_dataset()


    # 测试tfrecord数据文件没问题
    batch_size = 5
    img_loader = ImageLoader([global_configs.pic2attr_tfrecord_train_path], batch_size, 1, 40000)
    parsed_dataset = img_loader.launch_tfrecord_dataset()

    # 为解析出来的数据集弄一个迭代器
    iterator = parsed_dataset.make_one_shot_iterator()

    pic_name_batch, pic_class_batch, attr_label_batch, img_batch = iterator.get_next()

    pic_name_batch = tf.reshape(pic_name_batch, [batch_size])
    pic_class_batch = tf.reshape(pic_class_batch, [batch_size])


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        pic_name_batch_out, pic_class_batch_out, attr_label_batch_out, img_batch_out = sess.run([pic_name_batch, pic_class_batch, attr_label_batch, img_batch])

        print(pic_name_batch_out)
        print(pic_class_batch_out)
        print(attr_label_batch_out)
        print(img_batch_out)
        print(img_batch_out.shape)
        my_img = Image.fromarray(img_batch_out[1])
        my_img.show()

        coord.request_stop()
        coord.join(threads)