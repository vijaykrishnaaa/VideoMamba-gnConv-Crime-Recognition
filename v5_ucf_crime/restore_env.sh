#!/bin/bash
#PBS -N RESTORE_ENV
#PBS -q workq
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /home/y_rajeshwar/vlil_40/v5_ucf_crime/logs/restore_env_pbs.log

# Puts videomamba_env back at its ORIGINAL path from the 13 Oct backup, then checks that
# torch, OpenCV and the compiled Mamba kernels import on a GPU node.
# The original path matters: the env has hard-coded paths inside (scripts, and the editable
# mamba_ssm / causal_conv1d links to ~/videomamba_crime/Vim/...).
# Live progress:  tail -f ~/vlil_40/v5_ucf_crime/logs/restore_env_live.log

ENVS=/home/y_rajeshwar/anaconda3/envs
TAR=/home/y_rajeshwar/env_backup/videomamba_env_full.tar.gz
TMP=$ENVS/.videomamba_env_restore_tmp
LIVE=/home/y_rajeshwar/vlil_40/v5_ucf_crime/logs/restore_env_live.log
mkdir -p "$(dirname "$LIVE")"
exec >> "$LIVE" 2>&1          # everything below (including errors) goes to the live log

echo "===== JOB $PBS_JOBID on $(hostname) started $(date) ====="
if [ -e "$ENVS/videomamba_env" ]; then
  echo "videomamba_env already exists at $ENVS/videomamba_env: not unpacking, only checking it."
else
  if [ ! -f "$TAR" ]; then echo "ERROR: backup not found: $TAR"; exit 1; fi
  echo "Unpacking $TAR (5.9 GB) ..."
  rm -rf "$TMP"                # leftover of an earlier failed attempt of this job
  mkdir -p "$TMP"
  if ! tar -xzf "$TAR" -C "$TMP"; then
    echo "ERROR: unpack failed (disk full?). Partial files left in $TMP"
    exit 1
  fi
  mv "$TMP/videomamba_env" "$ENVS/videomamba_env" && rmdir "$TMP"
  echo "Unpacked to $ENVS/videomamba_env at $(date) ($(du -sh "$ENVS/videomamba_env" | cut -f1))"
fi

source /home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
conda activate videomamba_env || { echo "ERROR: conda cannot activate videomamba_env"; exit 1; }
echo "python: $(which python)"
python - <<'EOF'
import torch, torchvision, cv2, numpy
print('torch', torch.__version__, '| torchvision', torchvision.__version__,
      '| opencv', cv2.__version__, '| numpy', numpy.__version__)
ok = torch.cuda.is_available()
print('CUDA available:', ok, '|', torch.cuda.get_device_name(0) if ok else 'no GPU visible')
import causal_conv1d, mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn   # loads the compiled kernels
print('mamba_ssm     from', mamba_ssm.__file__)
print('causal_conv1d from', causal_conv1d.__file__)
if not ok:
    raise SystemExit('ERROR: torch cannot see the GPU')
print('ENV OK')
EOF
STATUS=$?
echo "===== EXIT CODE $STATUS at $(date) ====="
exit $STATUS
