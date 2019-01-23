'''
工具包
'''
import os
import tensorflow as tf
import numpy as np

def get_batch_image(root_path='./',batch_size=64,image_w=224,image_h=224):
    '''
    扫描相应路径下的图片文件，生成指定大小图片的批数据
    :param root_path: 文件根路径，在此路径下寻找所有的图片文件
    :param batch_size: 批大小
    :param image_w: 生成图像的宽度
    :param image_h: 生成图像的高度
    :return:
    '''
    image_paths = []
    labels = []
    if not os.path.exists(root_path):
        raise FileNotFoundError('root_path is not exists')

    pass

