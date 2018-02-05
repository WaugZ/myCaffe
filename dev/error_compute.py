# coding=utf-8

import caffe
import numpy as np
import os
import cv2

model_path = "/media/store/myAN/caffe_alexnet_train_iter_430000.caffemodel"
deploy_path = "/media/store/myAN/deploy.prototxt"
val_path = "/media/store/someDataSet/ILSVRC12_split/ILSVRC2012_img_val"
labels_file = "/media/store/myAN/label.txt"

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(deploy_path, model_path, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 117, 123]))
# transformer.set_raw_scale('data', 255)
# transformer.set_channel_swap('data', (2, 1, 0))

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

labels = np.loadtxt(labels_file, str, delimiter='\t')

total = 0
error = 0
for root, _, files in os.walk(val_path):
    for file in files:
        # img = caffe.io.load_image(os.path.join(root, file))
        cv_img = cv2.imread(os.path.join(root, file))
        net.blobs['data'].data[...] = transformer.preprocess('data', cv_img)
        net.forward()

        top_k = net.blobs['prob'].data[0].flatten().argsort()[::-1][:5]

        label_name = labels[top_k]

        true_label = os.path.basename(root)

        if true_label not in label_name:
            error += 1
            # print "Not good"
        else:
            # print "Bingo"
            pass
        total += 1
        if total % 1000 == 0:
            print "scaned %d pictures" % total
            print "Top 5 error rate is %.5f %%." % (100. * error / total)

print "Final top 5 error rate is %.5f %%." % (100. * error / total)
