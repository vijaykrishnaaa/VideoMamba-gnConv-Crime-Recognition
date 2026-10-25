#!/bin/bash
#PBS -N V6_PREP
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /home/y_rajeshwar/vlil_40/v6_crime_moments/logs/prep_pbs.log

# V6 step 1: cut every UCF-Crime crime video to its crime window (UCA timestamps for train,
# official annotation for test), store 32 frames per window, write splits/. CPU only.
# Live progress:  tail -f ~/vlil_40/v6_crime_moments/logs/prep_live.log

EXP=/home/y_rajeshwar/vlil_40/v6_crime_moments
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
conda activate videomamba_env || { echo "ERROR: conda env videomamba_env not found (run v5_ucf_crime/restore_env.sh)"; exit 2; }
cd /home/y_rajeshwar/vlil_40
export OMP_NUM_THREADS=1

python -u "$EXP/prepare_data_v6.py" --procs 4 --unmatched whole
STATUS=$?
echo "===== EXIT CODE $STATUS at $(date) ====="
exit $STATUS
