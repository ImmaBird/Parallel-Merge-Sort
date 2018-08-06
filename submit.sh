#!/bin/bash
make -C serial/
qsub serial/pbs_sort

make -C openmp/
qsub openmp/pbs_sort

make -C mpi/
qsub mpi/pbs_sort

make -C mpi-openmp/
qsub mpi-openmp/pbs_sort
