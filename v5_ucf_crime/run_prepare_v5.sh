#!/bin/bash
#PBS -N V5_PREP
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /home/y_rajeshwar/vlil_40/v5_ucf_crime/logs/prep_pbs.log

# V5 step 1: decode the 700 UCF-Crime videos of the official split ONCE into the frame cache
# and write splits/fold1..fold4/{train,val,test}.csv. CPU only.
# Live progress:  tail -f ~/vlil_40/v5_ucf_crime/logs/prep_live.log

EXP=/home/y_rajeshwar/vlil_40/v5_ucf_crime
LIVE=$EXP/logs/prep_live.log
CONDA_SH=/home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
mkdir -p $EXP/logs
exec >> "$LIVE" 2>&1          # everything below (including errors) goes to the live log

echo "===== JOB $PBS_JOBID on $(hostname) started $(date) ====="
if [ ! -f "$CONDA_SH" ]; then
  echo "ERROR: $CONDA_SH not found (Anaconda moved or removed?). Fix CONDA_SH in this script."
  exit 2
fi
source "$CONDA_SH"
conda activate videomamba_env || { echo "ERROR: conda env videomamba_env not found"; exit 2; }
cd /home/y_rajeshwar/vlil_40
export OMP_NUM_THREADS=1

python -u "$EXP/prepare_data_v5.py" --procs 8
STATUS=$?
echo "===== EXIT CODE $STATUS at $(date) ====="
exit $STATUS
