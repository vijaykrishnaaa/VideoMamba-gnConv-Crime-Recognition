#!/bin/bash
#PBS -N TEST_CRIME
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=01:00:00
#PBS -o crime_test_output.log
#PBS -e crime_test_error.log

cd /home/y_rajeshwar/vlil_40
source /home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
conda activate videomamba_env
python test_crime.py
