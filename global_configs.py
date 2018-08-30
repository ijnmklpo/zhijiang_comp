# -*- coding:utf-8 -*-
# 
# @Author: ijnmklpo
# @Time: 2018/8/19 上午2:33
# @Desc: 

import numpy as np
import os


root_path = '/Users/renyihao/Datas/zhijiang_cup/DatasetA_train_20180813'    # mac local
# root_path = 'D:/Datas/zhijiang_cup/DatasetA_train_20180813'    # win local

# root_path = '/data/ryh/datas/zhijiang_cup/DatasetA_train_20180813'          # remote

# 问题给出的数据
pic_fold_path = os.path.join(root_path, 'pics')
train_pic_class_path = os.path.join(root_path, 'train.txt')
attributes_per_class_path = os.path.join(root_path, 'attributes_per_class.txt')
class_wordembeddings_path = os.path.join(root_path, 'class_wordembeddings.txt')

# 清洗提取后的数据
class_attrlabel_path = os.path.join(root_path, 'class_attrlabel.txt')
train_pic_class_attr_path = os.path.join(root_path, 'pic_class_attrlabel.txt')
pic2attr_tfrecord_path = os.path.join(root_path, 'pic2attr_feats.tfrecord')

pic2attr_tfrecord_train_path = os.path.join(root_path, 'pic2attr_feats_train.tfrecord')
pic2attr_tfrecord_valid_path = os.path.join(root_path, 'pic2attr_feats_train.tfrecord')




if __name__ == '__main__':
    pass