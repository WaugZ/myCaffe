# coding=utf-8
import caffe
import os
import re

source_path = "/media/store/myAN"
solver_path = "/media/store/myAN/solver.prototxt"
snapshot_prefix = "/media/store/myAN/caffe_alexnet_train"
solverstate_path = ""

caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver(solver_path)

solverstates = [ss for ss in os.listdir(source_path) if ss.endswith(".solverstate")]
if not len(solverstates) == 0:
    latest_solver = ""
    latest_iter = 0
    for solverstate in solverstates:
        pattern = r"\d+"
        reg = re.findall(pattern, solverstate)
        i = int(reg[0])
        if i > latest_iter:
            latest_iter = i
            latest_solver = solverstate

    latest_solver = os.path.join(source_path, latest_solver)
    # print latest_solver

    solver.restore(latest_solver)

if os.path.isfile(solverstate_path) and solverstate_path.endswith(".solverstate"):
    solver.restore(solverstate_path)

while solver.iter < solver.param.max_iter:
    solver.step(1)
