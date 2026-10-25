#!/bin/bash
#PBS -N CRIME_V8_TRANSFER
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o /home/y_rajeshwar/vlil_40/v3_crime_transfer/crime_v8_output.log
#PBS -e /home/y_rajeshwar/vlil_40/v3_crime_transfer/crime_v8_error.log

cd /home/y_rajeshwar/vlil_40
source /home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
conda activate videomamba_env
python v3_crime_transfer/train_crime_v8.py
