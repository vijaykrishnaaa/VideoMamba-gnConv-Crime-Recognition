#!/bin/bash
#PBS -N V5_TRAIN
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /home/y_rajeshwar/vlil_40/v5_ucf_crime/logs/train_pbs.log

# V5 step 2: train + validate + final test on ONE fold of the official UCF-Crime split.
# (No ngpus=1: on this cluster that puts the job on Hold; the node's GPU is visible anyway.)
# Pick the fold and the starting checkpoint when submitting (defaults: FOLD=1, INIT=ucf101):
#   qsub -v FOLD=2,INIT=ucf101 run_train_v5.sh
# Live progress:  tail -f ~/vlil_40/v5_ucf_crime/runs/fold1_ucf101/logs/train_live.log
# If the job is killed (walltime / GPU taken), submit the same command again: it resumes from last.pth.

FOLD=${FOLD:-1}
INIT=${INIT:-ucf101}
EXP=/home/y_rajeshwar/vlil_40/v5_ucf_crime
RUN=$EXP/runs/fold${FOLD}_${INIT}
LIVE=$RUN/logs/train_live.log
CONDA_SH=/home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
mkdir -p $RUN/logs $EXP/logs
exec >> "$LIVE" 2>&1          # everything below (including errors) goes to the live log

echo "===== JOB $PBS_JOBID on $(hostname) started $(date) | FOLD=$FOLD INIT=$INIT ====="
nvidia-smi
if [ ! -f "$CONDA_SH" ]; then
  echo "ERROR: $CONDA_SH not found (Anaconda moved or removed?). Fix CONDA_SH in this script."
  exit 2
fi
source "$CONDA_SH"
conda activate videomamba_env || { echo "ERROR: conda env videomamba_env not found"; exit 2; }
cd /home/y_rajeshwar/vlil_40
export OMP_NUM_THREADS=4

python -u "$EXP/train_v5.py" --fold "$FOLD" --init "$INIT"
STATUS=$?
echo "===== EXIT CODE $STATUS at $(date) ====="
exit $STATUS
