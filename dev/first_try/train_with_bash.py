# coding=utf-8

import os, shutil, subprocess, random

data_path = '/media/store/myImplement/myCaffe/data/re'  # containing dir train and test
output_path = '/media/store/myImplement/myCaffe/dev/myfile'
caffe_root = '/media/store/myImplement/myCaffe/caffe'  # root path of compiled caffe

if not os.path.exists(output_path):
    os.makedirs(output_path)
if os.path.exists(os.path.join(output_path, 'train.txt')):
    os.remove(os.path.join(output_path, 'train.txt'))
if os.path.exists(os.path.join(output_path, 'test.txt')):
    os.remove(os.path.join(output_path, 'test.txt'))

print 'creating train.txt'
train_set = {}
for path, _, files in os.walk(os.path.join(data_path, 'train')):
    if len(files) > 0:
        random.shuffle(files)
    for file in files:
        image = os.path.join(path, file)
        label = str(int(file[0]) - 3)  # because the name of the pictures
        train_set[image] = label


with open(os.path.join(output_path, 'train.txt'), 'ab+') as open_file:
    # random.shuffle(train_set)
    for image in train_set:
        open_file.write("%s %s\n" % (image, train_set[image]))

print 'creating test.txt'
test_set = {}
for path, _, files in os.walk(os.path.join(data_path, 'test')):
    if len(files) > 0:
        random.shuffle(files)
    for file in files:
            image = os.path.join(path, file)
            label = str(int(file[0]) - 3)
            test_set[image] = label

with open(os.path.join(output_path, 'test.txt'), 'ab+') as open_file:
    # random.shuffle(train_set)
    for image in test_set:
        open_file.write("%s %s\n" % (image, test_set[image]))

if os.path.exists(os.path.join(output_path, 'img_train_lmdb')):
    shutil.rmtree(os.path.join(output_path, 'img_train_lmdb'))
if os.path.exists(os.path.join(output_path, 'img_test_lmdb')):
    shutil.rmtree(os.path.join(output_path, 'img_test_lmdb'))
convert_imageset = os.path.join(caffe_root, 'build', 'tools', 'convert_imageset')
print 'creating train_lmdb'
subprocess.call([convert_imageset, '--shuffle', '--resize_height=256', '--resize_width=256',
                 '/',
                 os.path.join(output_path, 'train.txt'), os.path.join(output_path, 'img_train_lmdb')])
print 'creating test_lmdb'
subprocess.call([convert_imageset, '--shuffle', '--resize_height=256', '--resize_width=256',
                 '/',
                 os.path.join(output_path, 'test.txt'), os.path.join(output_path, 'img_test_lmdb')])

raise Exception('Stop here')
print 'computing mean'
image_mean = os.path.join(caffe_root, 'build', 'tools', 'compute_image_mean')
subprocess.call([image_mean, os.path.join(output_path, 'img_train_lmdb'), os.path.join(output_path, 'mean.binaryproto')])

print 'preparing to train'
caffe = os.path.join(caffe_root, 'build', 'tools', 'caffe')
subprocess.call([caffe, 'train', '-solver', '/media/store/myImplement/myCaffe/dev/myfile/solver.prototxt'])
