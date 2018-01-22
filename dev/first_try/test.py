import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

t = caffe.Timer()