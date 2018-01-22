# coding=utf-8
# do almost the same as [file](train_with_bash.py) except implement with python
import sys
import caffe

solver_path = '/media/store/myImplement/myCaffe/dev/myfile/solver.prototxt'

caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver(solver_path)

while solver.iter != solver.param.max_iter:
    solver.step(1)

