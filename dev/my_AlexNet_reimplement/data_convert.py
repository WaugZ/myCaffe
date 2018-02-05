# coding=utf-8

import subprocess
import os
import shutil
import random
import create_lmdb as cl

data_path = "/media/store/someDataSet/ILSVRC12_split"
output_path = "/media/store/myANN"  # to save the train.txt and val.txt

if not os.path.exists(output_path):
    os.makedirs(output_path)


# 1. make a train.txt
# 1.1 train.txt contains path to image and its label
# 1.2 maybe a label map is a good idea
def create_txt():
    print "cleaning label.txt, train.txt, val.txt"
    for file in ['label.txt', 'train.txt', 'val.txt']:
        if os.path.isfile(os.path.join(output_path, file)):
            os.remove(os.path.join(output_path, file))

    print("writing label.txt")
    labels = {}
    with open(os.path.join(output_path, 'label.txt'), 'ab+') as label_file:
        lable_ind = 0
        for path, subdirs, files in os.walk(os.path.join(data_path, "ILSVRC2012_img_val")):
            for subdir in subdirs:
                if subdir not in labels:
                    labels[subdir] = lable_ind
                    lable_ind += 1
                    label_file.write(subdir + '\t' + str(lable_ind) + '\n')
    print len(labels)

    print "writing train.txt"
    train_set = {}
    with open(os.path.join(output_path, 'train.txt'), 'ab+') as train_file:
        for path, subdirs, files in os.walk(os.path.join(data_path, "ILSVRC2012_img_train")):
            for file in files:
                image = os.path.join(path, file)
                label_name = os.path.dirname(path).split('/')[-1]
                label = str(labels[label_name])
                train_set[image] = label
                # train_file.write(image + ' ' + label + '\n')
        # print len(labels)
        images = train_set.keys()
        random.shuffle(images)
        for image in images:
            train_file.write("%s %s\n" % (image, train_set[image]))

    print "writing val.txt"
    with open(os.path.join(output_path, 'val.txt'), 'ab+') as val_file:
        val_set = {}
        for path, subdirs, files in os.walk(os.path.join(data_path, "ILSVRC2012_img_val")):
            for file in files:
                image = os.path.join(path, file)
                label_name = os.path.basename(path)
                label = str(labels[label_name])
                val_set[image] = label
                # val_file.write(image + ' ' + label + '\n')
        images = val_set.keys()
        random.shuffle(images)
        for image in images:
            val_file.write("%s %s\n" % (image, val_set[image]))


# 2. change original data to lmdb using shell bash conver_imageset
# 2.1 which need train.txt , test.txt and so on
# (option) maybe need long time :(
def create_lmdb():
    caffe_root = "/media/store/myImplement/myCaffe/caffe"
    convert_path = os.path.join(caffe_root, 'build', 'tools', 'convert_imageset')
    if not os.path.exists(os.path.join(output_path, "img_val_lmdb")):
        print "creating val.lmdb"
        subprocess.call([convert_path, '--shuffle', "/", '--resize_height=256', '--resize_width=256',
                         os.path.join(output_path, 'val.txt'), os.path.join(output_path, 'img_val_lmdb')])

    if not os.path.exists(os.path.join(output_path, "img_train_lmdb")):
        print "creating train.lmdb"
        subprocess.call([convert_path, '--shuffle', "/", '--resize_height=256', '--resize_width=256',
                         os.path.join(output_path, 'train.txt'), os.path.join(output_path, 'img_train_lmdb')])


if __name__ == "__main__":
    create_txt()
    # create_lmdb()
    train_txt = os.path.join(output_path, 'train.txt')
    val_txt = os.path.join(output_path, 'val.txt')
    train_lmdb = os.path.join(output_path, "img_train_lmdb")
    val_lmdb = os.path.join(output_path, "img_val_lmdb")
    cl.create_lmdb(train_txt_path=train_txt, train_lmdb_output_path=train_lmdb, val_txt_path=val_txt,
                val_lmdb_output_path=val_lmdb, new_height=256, new_width=256, SHUFFLE=False)
