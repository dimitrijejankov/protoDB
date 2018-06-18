#!/usr/bin/env python

from subprocess import call
from subprocess import STDOUT

cores = [1, 2, 4, 8]
sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800]


def initialize():

    # write backs completed
    target = open("results_matrix_multiply.csv", 'w')
    target.truncate()
    target.close()


def run_experiment(c, s):

    print("Running the experiment with : cores=%s, sizes=%s" % (c, s))

    # run the cmake
    call(["./test-matrix-multiply", str(s), str(c)], stderr=STDOUT)


# initialize the results files
initialize()

# run the sync experiment
for core in cores:
    for size in sizes:
        run_experiment(core, size)
