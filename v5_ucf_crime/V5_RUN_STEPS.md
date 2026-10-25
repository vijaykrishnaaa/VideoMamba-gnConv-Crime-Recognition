# V5: UCF-Crime, how to run it

## The plan in short

| Item | Choice |
|---|---|
| Dataset | UCF-Crime, official **Action Recognition** split (`Action_Regnition_splits`): 13 crime classes + Normal = **14 classes × 50 videos = 700 videos** |
| Folds | 4 official folds. Each fold: 532 train / 168 test (38 / 12 per class) |
| Train / val / test | Per fold: **448 train / 84 val / 168 test**. Val (6 per class) is taken from the official train list and picks the best epoch. Test is the official list, used once at the end |
| Starting checkpoint | `v1_ucf101/checkpoints_ucf101/best_ucf101_v6.pth` (K400 → UCF-101), same start as V4. gnConv in layers 16-23 starts at zero |
| Optional comparison | `INIT=v4` starts from V4's `best_model_v9.pth` (K400 → UCF-101 → violence) |
| Model / recipe | Exactly V4: `mamba_simple_v7.py`, layers 16-23 trained, same LRs, regularisation, EMA, float32 |
| Number to report | Mean test accuracy over the 4 folds (that is how the paper reports C3D 23.0% and TCNN 28.4%) |
| Extra number | Crime-window test: test videos that have a temporal annotation (36 / 38 / 31 / 14 in folds 1-4) are also tested on 16 frames from inside the annotated crime. Pooled over the 4 folds. Training never uses the annotations |

Not used: `Anomaly_Detection_splits` (the other task: normal vs abnormal, with all 1,900 videos).

## Files (upload this whole folder)

| File | What it does |
|---|---|
| `prepare_data_v5.py` | Finds the 700 videos, stores 64 frames per video (+16 crime-window frames where annotated) in `frame_cache/`, writes `splits/fold1..4/` |
| `run_prepare_v5.sh` | PBS job for the prep (CPU, 8 cores) |
| `train_v5.py` | Training + validation + one final test, for one fold |
| `run_train_v5.sh` | PBS job for training (`-v FOLD=..,INIT=..`) |
| `mamba_simple_v7.py` | The fixed V4 backbone (unchanged copy) |
| `restore_env.sh` | PBS job: puts `videomamba_env` back from `~/env_backup` and checks it imports |

## Step 0: quick checks (PuTTY, login node; these only list files)

```bash
ls ~/anaconda3/etc/profile.d/conda.sh
ls ~/anaconda3/envs | grep videomamba
ls ~/final_year_project_29/ucf_dataset/UCF_Crimes-Train-Test-Split/Action_Regnition_splits
ls -lh ~/vlil_40/v1_ucf101/checkpoints_ucf101/best_ucf101_v6.pth
df -h ~
```

On 17 Oct 2026 `videomamba_env` was gone from `~/anaconda3/envs` (base Anaconda was still there).
`restore_env.sh` unpacks `~/env_backup/videomamba_env_full.tar.gz` back to `~/anaconda3/envs/videomamba_env`
(the original path) and checks that torch, OpenCV and the Mamba kernels import on a GPU node.
It skips the unpack if the env already exists, so it is safe to submit again.

## Step 1: upload (WinSCP)

Left panel: `C:\Users\LENOVO\Desktop\Vlil40\Current work\`
Right panel: `/home/y_rajeshwar/vlil_40/`
Drag the folder `v5_ucf_crime` across. You should get `/home/y_rajeshwar/vlil_40/v5_ucf_crime/`.

## Step 2: fix Windows line endings (PuTTY)

```bash
cd ~/vlil_40/v5_ucf_crime
sed -i 's/\r$//' *.py *.sh
ls -l
```

## Step 3: submit env restore -> prep -> training as one chain

```bash
cd ~/vlil_40/v5_ucf_crime
ENV=$(qsub restore_env.sh); echo $ENV
PREP=$(qsub -W depend=afterok:$ENV run_prepare_v5.sh); echo $PREP
TRAIN=$(qsub -W depend=afterok:$PREP -v FOLD=1,INIT=ucf101 run_train_v5.sh); echo $TRAIN
qstat -u y_rajeshwar
```

Each job starts only when the one before it finished without errors. Waiting jobs show **H**: that is
the dependency, not the old ngpus problem. If the env check fails, prep and training never start.
(Once the env exists, later runs only need the PREP/TRAIN lines, or just TRAIN.)

## Step 4: watch live (Ctrl+C stops watching, not the job)

```bash
tail -f ~/vlil_40/v5_ucf_crime/logs/restore_env_live.log
tail -f ~/vlil_40/v5_ucf_crime/logs/prep_live.log
cat ~/vlil_40/v5_ucf_crime/logs/prep_report.txt
tail -f ~/vlil_40/v5_ucf_crime/runs/fold1_ucf101/logs/train_live.log
grep "Epoch" ~/vlil_40/v5_ucf_crime/runs/fold1_ucf101/logs/train_live.log
```

## Step 5: after fold 1, run folds 2-4 one after another

```bash
cd ~/vlil_40/v5_ucf_crime
J2=$(qsub -v FOLD=2 run_train_v5.sh); echo $J2
J3=$(qsub -W depend=afterany:$J2 -v FOLD=3 run_train_v5.sh); echo $J3
J4=$(qsub -W depend=afterany:$J3 -v FOLD=4 run_train_v5.sh); echo $J4
cat ~/vlil_40/v5_ucf_crime/results_all_runs.csv
```

They run one at a time so two jobs never share one V100.

## If a job is killed

Submit the same `qsub ... run_train_v5.sh` command again. It resumes from `runs/foldK_INIT/checkpoints/last.pth`.

## Where results go

`runs/fold1_ucf101/logs/train_live.log`, `logs/history.csv`, `results/` (test_results.json,
confusion_matrix.csv, per_class_metrics.csv, test_predictions.csv, crime_window_predictions.csv),
and `results_all_runs.csv` (one row per finished fold; the end of each log prints the running
4-fold mean and the pooled crime-window numbers).
