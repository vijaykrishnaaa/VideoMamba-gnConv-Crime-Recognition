#!/bin/bash
#PBS -N V4_PREP
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=06:00:00
#PBS -o /home/y_rajeshwar/vlil_40/v4_crime_fixed/logs/prep_pbs_output.log
#PBS -e /home/y_rajeshwar/vlil_40/v4_crime_fixed/logs/prep_pbs_error.log

# V4 step 1: decode every video once into the frame cache + write train/val/test lists.
# CPU only. Live progress:  tail -f ~/vlil_40/v4_crime_fixed/logs/prep_live.log

EXP=/home/y_rajeshwar/vlil_40/v4_crime_fixed
LIVE=$EXP/logs/prep_live.log
mkdir -p $EXP/logs

cd /home/y_rajeshwar/vlil_40
source /home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
conda activate videomamba_env
export OMP_NUM_THREADS=1

echo "===== JOB $PBS_JOBID on $(hostname) started $(date) =====" >> "$LIVE"
python -u "$EXP/prepare_data_v9.py" --procs 4 >> "$LIVE" 2>&1
STATUS=$?
echo "===== EXIT CODE $STATUS at $(date) =====" >> "$LIVE"
exit $STATUS
