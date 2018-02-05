# coding=utf-8

import subprocess
import os
import shutil
import random
import create_lmdb

data_path = "/media/store/someDataSet/ILSVRC12_split"
output_path = "/media/store/myAN"  # to save the train.txt and val.txt

NEED_VAL = True
NEED_LMDB = True

if not os.path.exists(output_path):
    os.makedirs(output_path)


# 1. make a train.txt
# 1.1 train.txt contains path to image and its label
# 1.2 maybe a label map is a good idea
def create_txt(TRAIN=True, VAL=True):
    assert type(TRAIN) == bool and type(VAL) == bool

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

    if TRAIN:
        print "writing train.txt"
        train_set = {}
        with open(os.path.join(output_path, 'train.txt'), 'ab+') as train_file:
            for path, subdirs, files in os.walk(os.path.join(data_path, "ILSVRC2012_img_train")):
                for file in files:
                    image = os.path.join(path, file)
                    label_name = os.path.dirname(path).split('/')[-1]
                    label = str(labels[label_name])
                    train_set[image] = label
            images = train_set.keys()
            # shuffle train_set by shuffling the key
            random.shuffle(images)
            for image in images:
                train_file.write("%s %s\n" % (image, train_set[image]))

    if VAL:
        print "writing val.txt"
        with open(os.path.join(output_path, 'val.txt'), 'ab+') as val_file:
            val_set = {}
            for path, subdirs, files in os.walk(os.path.join(data_path, "ILSVRC2012_img_val")):
                for file in files:
                    image = os.path.join(path, file)
                    label_name = os.path.basename(path)
                    label = str(labels[label_name])
                    val_set[image] = label
            images = val_set.keys()
            # shuffle val_set by shuffling the key
            random.shuffle(images)
            for image in images:
                val_file.write("%s %s\n" % (image, val_set[image]))


if __name__ == "__main__":
    create_txt(VAL=NEED_VAL)
    if NEED_LMDB:
        train_txt = os.path.join(output_path, 'train.txt')
        val_txt = os.path.join(output_path, 'val.txt')
        train_lmdb = os.path.join(output_path, "img_train_lmdb")
        val_lmdb = os.path.join(output_path, "img_val_lmdb")
        create_lmdb.create_lmdb(train_txt_path=train_txt, train_lmdb_output_path=train_lmdb, val_txt_path=val_txt,
                       val_lmdb_output_path=val_lmdb, new_height=256, new_width=256, SHUFFLE=False)
