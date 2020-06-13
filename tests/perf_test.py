#!/usr/bin/env python
import timeit
import tabulate
import numpy as np
from numpy.linalg import norm

import iops_pyx
import iops_cpp
import iops_sci

class PerfExperiment2D(object):
    def __init__(self):
        self.a = np.array([0.5, 0.00], dtype=np.float32)
        self.b = np.array([0.0, 0.25], dtype=np.float32)
        self.p_off = np.array([1.0, 2.0], dtype=np.float32)
        self.p_on = (self.a + self.b) / 2.0; # center of mass for pOnElement
        self.n_p_off = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        ab = self.b - self.a
        self.n_p_on = np.array([-ab[1], ab[0]], dtype=np.float32)
        self.n_p_on = self.n_p_on / norm(self.n_p_on)
        self.table_l = []
        self.table_m = []
        self.table_mt = []
        self.table_n = []

    def l_experiments(self, module):
        row = [module.__name__]
        if module.__name__ == "iops_sci":
            row.append(timeit.timeit(lambda: module.l_2d_off_k0(self.p_off, self.a, self.b), number=1000))
        else:
            row.append(timeit.timeit(lambda: module.l_2d(0.0, self.p_off, self.a, self.b, False), number=1000))
        row.append(timeit.timeit(lambda: module.l_2d(16.0, self.p_off, self.a, self.b, False), number=1000))
        row.append(timeit.timeit(lambda: module.l_2d(0.0, self.p_on, self.a, self.b, True), number=1000))
        row.append(timeit.timeit(lambda: module.l_2d(16.0, self.p_on, self.a, self.b, True), number=1000))
        self.table_l.append(row)

    def m_experiments(self, module):
        row = [module.__name__]
        row.append(timeit.timeit(lambda: module.m_2d(0.0, self.p_off, self.a, self.b, False), number=1000))
        row.append(timeit.timeit(lambda: module.m_2d(16.0, self.p_off, self.a, self.b, False), number=1000))
        row.append(timeit.timeit(lambda: module.m_2d(0.0, self.p_on, self.a, self.b, True), number=1000))
        row.append(timeit.timeit(lambda: module.m_2d(16.0, self.p_on, self.a, self.b, True), number=1000))
        self.table_m.append(row)

    def mt_experiments(self, module):
        row = [module.__name__]
        row.append(timeit.timeit(lambda: module.mt_2d(0.0, self.p_off, self.n_p_off,
                                                                  self.a, self.b, False), number=1000))
        row.append(timeit.timeit(lambda: module.mt_2d(16.0, self.p_off, self.n_p_off,
                                                                  self.a, self.b, False), number=1000))
        row.append(timeit.timeit(lambda: module.mt_2d(0.0, self.p_on, self.n_p_on,
                                                                  self.a, self.b, True), number=1000))
        row.append(timeit.timeit(lambda: module.mt_2d(16.0, self.p_on, self.n_p_on,
                                                                  self.a, self.b, True), number=1000))
        self.table_mt.append(row)

    def n_experiments(self, module):
        row = [module.__name__]
        row.append(timeit.timeit(lambda: module.n_2d(0.0, self.p_off, self.n_p_off,
                                                                self.a, self.b, False), number=1000))
        row.append(timeit.timeit(lambda: module.n_2d(16.0, self.p_off, self.n_p_off,
                                                                self.a, self.b, False), number=1000))
        row.append(timeit.timeit(lambda: module.n_2d(0.0, self.p_on, self.n_p_on,
                                                                self.a, self.b, True), number=1000))
        row.append(timeit.timeit(lambda: module.n_2d(16.0, self.p_on, self.n_p_on,
                                                                self.a, self.b, True), number=1000))
        self.table_n.append(row)

    def run(self):
        modules = [iops_pyx, iops_cpp, iops_sci]

        for module in modules:
            self.l_experiments(module)
            self.m_experiments(module)
            self.mt_experiments(module)
            self.n_experiments(module)

if __name__ == "__main__":
    experiment = PerfExperiment2D()
    experiment.run()
    base_header = ["k=0, off", "k=16, off", "k=0, on", "k=16, on"]
    print(tabulate.tabulate(experiment.table_l, headers=["l_2d"] + base_header) + "\n")
    print(tabulate.tabulate(experiment.table_m, headers=["m_2d"] + base_header) + "\n")
    print(tabulate.tabulate(experiment.table_mt, headers=["mt_2d"] + base_header) + "\n")
    print(tabulate.tabulate(experiment.table_n, headers=["n_2d"] + base_header) + "\n")
