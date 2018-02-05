# coding=utf-8
import os
import glob
import random
import numpy as np
import shutil
import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb


def create_lmdb(train_txt_path, train_lmdb_output_path, val_txt_path="", val_lmdb_output_path="",
                new_width=256, new_height=256, SHUFFLE=True):
    # verification
    assert os.path.isfile(train_txt_path), "{} does not exist".format(train_txt_path)
    if val_txt_path != "":
        assert os.path.isfile(val_txt_path), "{} does not exist".format(val_txt_path)
    assert type(train_lmdb_output_path) == str, "train_lmdb_output_path is not a path"
    if val_lmdb_output_path != "":
        assert type(val_lmdb_output_path) == str, "val_lmdb_output_path is not a path"
    assert type(new_width) == int and type(new_height) == int and new_width > 0 and new_height > 0, \
        "new_width and new_height should be positive integer"
    assert type(SHUFFLE) == bool, "SHUFFLE should be bool"

    # Size of images
    IMAGE_WIDTH = new_width
    IMAGE_HEIGHT = new_height

    # path to train.txt, val.txt
    train_path = train_txt_path
    val_path = val_txt_path

    # path to train_lmdb, validation_lmdb(output path)
    train_lmdb = train_lmdb_output_path
    val_lmdb = val_lmdb_output_path

    # 如果存在了这个文件夹, 先删除
    if os.path.exists(train_lmdb):
        shutil.rmtree(train_lmdb)
    if os.path.exists(val_lmdb):
        shutil.rmtree(val_lmdb)

    # 读取图像
    train_data = np.loadtxt(train_path, dtype=str, delimiter=' ')
    if val_path != "":
        test_data = np.loadtxt(val_path, dtype=str, delimiter=' ')

    # Shuffle train_data
    if SHUFFLE:
        random.shuffle(train_data)

    # 图像的变换, 直方图均衡化, 以及裁剪到 IMAGE_WIDTH x IMAGE_HEIGHT 的大小
    def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
        #Histogram Equalization
        # img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        # img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        # img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

        #Image Resizing, 三次插值
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        return img

    def make_datum(img, label):
        #image is numpy.ndarray format. BGR instead of RGB
        return caffe_pb2.Datum(
            channels=3,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            label=label,
            data=np.rollaxis(img, 2).tobytes())  # or .tostring() if numpy < 1.9

    # 打开 lmdb 环境, 生成一个数据文件，定义最大空间, 1e12 = 1000000000000.0
    in_db = lmdb.open(train_lmdb, map_size=int(1e12))
    in_txn = in_db.begin(write=True)  # 创建操作数据库句柄
    for in_idx, img_label in enumerate(train_data):
        # 读取图像. 做直方图均衡化、裁剪操作
        img_path = img_label[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        label = int(img_label[1])
        # if 'cat' in img_path: # 组织 label, 这里是如果文件名称中有 'cat', 标签就是 0
        #     label = 0         # 如果图像名称中没有 'cat', 有的是 'dog', 标签则为 1
        # else:                 # 这里方, label 需要自己去组织
        #     label = 1         # 每次情况可能不一样, 灵活点
        # img = img[:, :, ::-1]
        # img = img.transpose((2, 0, 1))
        datum = make_datum(img, label)
        # '{:0>5d}'.format(in_idx):
        #      lmdb的每一个数据都是由键值对构成的,
        #      因此生成一个用递增顺序排列的定长唯一的key
        in_txn.put('{:0>8d}'.format(in_idx), datum.SerializeToString())  # 调用句柄，写入内存
        if in_idx % 1000 == 0:
            in_txn.commit()
            in_txn = in_db.begin(write=True)
            print "processed {:d} train pictures".format(in_idx)
    print "Finish processing all {:d} training pictures".format(in_idx + 1)

    # 结束后记住释放资源，否则下次用的时候打不开。。。
    in_db.close()

    if val_path != "":
        # 创建验证集 lmdb 格式文件
        print '\nCreating validation_lmdb'
        in_db = lmdb.open(val_lmdb, map_size=int(1e12))
        in_txn = in_db.begin(write=True)
        for in_idx, img_label in enumerate(test_data):
            img_path = img_label[0]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            label = int(img_label[1])
            # if 'cat' in img_path:
            #     label = 0
            # else:
            #     label = 1
            # img = img[:, :, ::-1]
            # img = img.transpose((2, 0, 1))
            datum = make_datum(img, label)
            in_txn.put('{:0>8d}'.format(in_idx), datum.SerializeToString())
            if in_idx % 1000 == 0:
                in_txn.commit()
                in_txn = in_db.begin(write=True)
                print "processed {:d} val pictures".format(in_idx)
        in_db.close()
    print "Finish processing all {:d} val pictures".format(in_idx + 1)


if __name__ == "__main__":
    train_txt = "/media/store/myImplement/myCaffe/dev/myfile/train.txt"
    val_txt = "/media/store/myImplement/myCaffe/dev/myfile/test.txt"
    train_lmdb = "/media/store/myImplement/myCaffe/dev/myfile/img_train_lmdb"
    val_lmdb = "/media/store/myImplement/myCaffe/dev/myfile/img_test_lmdb"
    create_lmdb(train_txt_path=train_txt, train_lmdb_output_path=train_lmdb, val_txt_path=val_txt,
                val_lmdb_output_path=val_lmdb, new_height=256, new_width=256)

