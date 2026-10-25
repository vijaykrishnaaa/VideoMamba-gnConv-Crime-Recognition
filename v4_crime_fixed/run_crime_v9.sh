#!/bin/bash
#PBS -N CRIME_V9
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o /home/y_rajeshwar/vlil_40/v4_crime_fixed/logs/train_pbs_output.log
#PBS -e /home/y_rajeshwar/vlil_40/v4_crime_fixed/logs/train_pbs_error.log

# V4 step 2: train + validate + final test.
# (No ngpus=1: on this cluster that puts the job on Hold; the node's GPU is visible anyway.)
# Live progress:  tail -f ~/vlil_40/v4_crime_fixed/logs/train_live.log
# If the job is killed (walltime / GPU taken), just qsub this file again: it resumes from last_v9.pth.

EXP=/home/y_rajeshwar/vlil_40/v4_crime_fixed
LIVE=$EXP/logs/train_live.log
mkdir -p $EXP/logs

cd /home/y_rajeshwar/vlil_40
source /home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
conda activate videomamba_env
export OMP_NUM_THREADS=4

{
  echo "===== JOB $PBS_JOBID on $(hostname) started $(date) ====="
  nvidia-smi
} >> "$LIVE" 2>&1
python -u "$EXP/train_crime_v9.py" >> "$LIVE" 2>&1
STATUS=$?
echo "===== EXIT CODE $STATUS at $(date) =====" >> "$LIVE"
exit $STATUS
