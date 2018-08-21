# -*- coding:utf-8 -*-
# 
# @Author: ijnmklpo
# @Time: 2018/8/19 上午2:33
# @Desc: 

import numpy as np
import os


root_path = '/Users/renyihao/Datas/zhijiang_cup/DatasetA_train_20180813'    # local
# root_path = '/data/ryh/datas/zhijiang_cup/DatasetA_train_20180813'          # remote

# 问题给出的数据
pic_fold_path = os.path.join(root_path, 'pics')
train_pic_class_path = os.path.join(root_path, 'train.txt')
attributes_per_class_path = os.path.join(root_path, 'attributes_per_class.txt')
class_wordembeddings_path = os.path.join(root_path, 'class_wordembeddings.txt')

# 清洗提取后的数据
class_attrlabel_path = os.path.join(root_path, 'class_attrlabel.txt')



if __name__ == '__main__':
    pass