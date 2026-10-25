#!/bin/bash
#PBS -N UCF101_V6
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=24:00:00
#PBS -o ucf101_v6_train_output.log
#PBS -e ucf101_v6_train_error.log

cd /home/y_rajeshwar/vlil_40
source /home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
conda activate videomamba_env
python train_ucf101_v6.py
