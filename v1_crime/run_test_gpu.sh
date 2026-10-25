#!/bin/bash
#PBS -N TEST_GPU_CRIME
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=00:15:00
#PBS -o ~/crime_test_gpu_out.log
#PBS -e ~/crime_test_gpu_err.log

cd /home/y_rajeshwar/vlil_40
source /home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
conda activate videomamba_env
python test_crime_gpu.py
