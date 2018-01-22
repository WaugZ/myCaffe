# coding=utf-8
import caffe
import os

source_path = "/media/store/myAN"
solver_path = "/media/store/myAN/solver.prototxt"

caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver(solver_path)

while solver.iter < solver.param.max_iter:
    solver.step(1)
