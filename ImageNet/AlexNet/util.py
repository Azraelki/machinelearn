'''
工具包
'''
import os
import tensorflow as tf
import numpy as np
import tarfile
import pyprind
import pandas as pd
import matplotlib.pyplot as plt

def get_image_csv(root_path='./'):
    '''
    扫描根目录下的所有文件夹，以每个文件夹为标签，文件夹下面的图片为数据
    :param root_path: 文件根路径，在此路径下寻找所有的图片文件
    :return:
    '''
    image_paths = []
    labels = []
    if not os.path.exists(root_path):
        raise FileNotFoundError('root_path is not exists')
    dirnames = os.listdir(root_path)
    label_count = 0;
    for dir in dirnames:
        if os.path.isdir(os.path.join(root_path,dir)):
            label_count += 1
            filenames = np.array(os.listdir(os.path.join(root_path,dir)))
            labels = np.ones(len(filenames))*label_count
            data = np.row_stack((filenames,labels)).T
            df = pd.DataFrame(data,columns=['imagePath', 'label'])
            df['imagePath'] = df['imagePath'].apply(lambda x:os.path.join(root_path,dir,x))
            df.to_csv('./imageData.csv',mode='a',encoding='utf-8',index=False,header=False)
            data = np.array([[dir,label_count]])
            df = pd.DataFrame(data,columns=['labelName', 'label'])
            df.to_csv('./labelNameMap.csv',mode='a',encoding='utf-8',index=False,header=False)

def decompress_file(root_path='',suffix='.tar'):
    '''
    将指定目录下压缩文件解压到压缩文件名的目录下
    :param root_path: 根目录
    :param suffix: 文件后缀名
    :return:
    '''
    if not os.path.exists(root_path):
        raise FileNotFoundError('root_path is not exists')
    files = os.listdir(root_path)
    files = [i for i in files if i.endswith(suffix)]
    bar = pyprind.ProgBar(len(files))
    if suffix == '.tar':
        for file in files:
            source = os.path.join(root_path,file)
            target_dir = os.path.join(root_path,file.replace(suffix,''))
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            tar = tarfile.open(source,mode='r')
            tar.extractall(path=target_dir)
            tar.close()
            bar.update()

def show_img(title,image):
    # 显示图片
    plt.imshow(image)
    plt.axis('on')
    plt.title(title)
    plt.show()


width = 227
height = 227


def show_image(title, image):
    '''
    显示图片
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()


def tf_read_image(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    if width > 0 and height > 0:
        image = tf.image.resize_images(image, [height, width])
    image = tf.cast(image, tf.float32) * (1. / 255.0)  # 归一化
    return image, label


def get_dataset(files_list, labels_list, epoch=10,batch_size=64, shuffle=True):
    '''
    :param files_list:
    :param labels_list:
    :param batch_size:
    :param shuffle:
    :return:
    '''
    # 构建数据集
    dataset = tf.data.Dataset.from_tensor_slices((files_list, labels_list))
    if shuffle:
        dataset = dataset.shuffle(files_list.shape[0])
    dataset = dataset.repeat(epoch)  # 空为无限循环
    dataset = dataset.map(tf_read_image, num_parallel_calls=2)  # num_parallel_calls一般设置为cpu内核数量
    dataset = dataset.batch(batch_size)
    return dataset




if __name__ == '__main__':
    # get_batch_image('I:\imagenet\ILSVRC2012_img_train')
    # get_image_csv('F:\ASSDsoftware\work\program\pycharmworkspace\machinelearndata\\alexnet\\train')
    df = pd.read_csv('./imageData.csv')

    dataset = get_dataset(df['imagePath'].values,df['label'].values,batch_size=1000,epoch=1)
    max_iterate = 100
    with tf.Session() as sess:
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.make_initializer(dataset)
        sess.run(init_op)
        iterator = iterator.get_next()
        for i in range(max_iterate):
            images,labels = sess.run(iterator)
            print('shape:{}'.format(images.shape))






