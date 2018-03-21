#!/bin/sh
#PBS -l walltime=36:00:00
#PBS -N dense_cate3
#PBS -q large
#PBS -l nodes=1

# Tensorflow example using virtualenv
#
# Author: Richard Berger
#
# For questions please contact <hpc@temple.edu>
cd $PBS_O_WORKDIR

module load python
module load numpy
module load scipy
module load cuda
module load cudnn

# GPU version
source gpu_env_py3/bin/activate


python densenet_cate3.py > log_dense_cate3

deactivate
