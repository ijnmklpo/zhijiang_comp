# -*- coding:utf-8 -*-
# 
# @Author: ijnmklpo
# @Time: 2018/8/19 上午2:28
# @Desc: 数据制作。要把图片信息和属性的特征信息（作为标签）拼在一起，

import numpy as np
import global_configs
import pandas as pd


if __name__ == '__main__':
    train_pic_class_df = pd.read_csv(global_configs.train_pic_class_path, sep='\t')
    train_pic_class_df.columns = ['pic_name', 'class']
    class_attributes_df = pd.read_csv(global_configs.attributes_per_class_path, sep='\t')
    class_attributes_df.columns = ['class',
                                    'attr1', 'attr2', 'attr3', 'attr4', 'attr5',
                                    'attr6', 'attr7', 'attr8','attr9', 'attr10',
                                    'attr11', 'attr12', 'attr13', 'attr14', 'attr15',
                                    'attr16', 'attr17', 'attr18', 'attr19', 'attr20',
                                    'attr21', 'attr22', 'attr23', 'attr24', 'attr25',
                                    'attr26', 'attr27', 'attr28', 'attr29', 'attr30',
                                    ]

    def merge_attributes_to_label(x):
        label = [x['attr1'], x['attr2'], x['attr3'], x['attr4'], x['attr5'],
                 x['attr6'], x['attr7'], x['attr8'], x['attr9'], x['attr10'],
                 x['attr11'], x['attr12'], x['attr13'], x['attr14'], x['attr15'],
                 x['attr16'], x['attr17'], x['attr18'], x['attr19'], x['attr20'],
                 x['attr21'], x['attr22'], x['attr23'], x['attr24'], x['attr25'],
                 x['attr26'], x['attr27'], x['attr28'], x['attr29'], x['attr30'],
                ]
        label = [float(item) for item in label]
        return label


    attr_label = class_attributes_df.apply(lambda x: merge_attributes_to_label(x), axis=1)
    attr_label = attr_label.to_frame()
    attr_label.columns = ['attr_label']
    pic_attr_label_df = pd.concat([class_attributes_df, attr_label], axis=1)
    print(pic_attr_label_df.head())
    print(pic_attr_label_df.shape)

    class_attrlabel = pic_attr_label_df[['class', 'attr_label']]

    class_attrlabel.to_csv(global_configs.class_attrlabel_path, index=False, sep='\t')

    # print(train_pic_class_df.head)
    # print(class_attributes_df.head)