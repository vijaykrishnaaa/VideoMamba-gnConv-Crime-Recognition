# V6: UCF-Crime crime moments, how to run it

## The plan in short

| Item | Choice |
|---|---|
| Question | "A crime is happening right now: which one?" (what the robot needs) |
| Classes | 13 crimes (no Normal: deciding *whether* something is a crime is the detector's job) |
| Test | The 140 official UCF-Crime test crime videos, cut to their **official** crime window (`Temporal_Anomaly_Annotation.txt`) |
| Train / val | All other crime videos (~810), cut to the crime window from **UCA** timestamps (CVPR 2024). 15% of them is validation |
| How the train window is found | UCA describes every event with start/end times but does not say which one is the crime, so crime sentences are picked by keywords per class (`CRIME_WORDS` in `prepare_data_v6.py`). Checked: 816 of 944 videos get a window; on the 124 official test videos it covers 70% of the true crime window. Videos without a match use the whole video |
| Frames | 32 frames stored inside each window; training takes 8, the test looks at all 32 (4 clips x 3 crops) |
| Model / recipe | Same as V5 (VideoMamba-S + gnConv in layers 16-23, UCF-101 start, same LRs and regularisation) |
| Runs | 3 seeds (42, 7, 123) on the same split; report the mean |
| Numbers to report | Accuracy, balanced accuracy (every class counts the same; the test has 2 Abuse but 23 Shooting videos), macro-F1 |

V5 stays as the standard-benchmark result (37.05%, whole videos, 14 classes). V6 is the realistic
"crime moment" result. They answer different questions, so both go in the report.

## Files (upload this whole folder)

| File | What it does |
|---|---|
| `prepare_data_v6.py` | Finds the crime windows, stores 32 frames per window in `frame_cache/`, writes `splits/` and `logs/prep_report.txt` |
| `run_prepare_v6.sh` | PBS job for the prep (CPU, 4 cores) |
| `train_v6.py` | Training + validation + one final test |
| `run_train_v6.sh` | PBS job for training (`-v SEED=..,INIT=..`) |
| `mamba_simple_v7.py` | The fixed V4 backbone (unchanged copy) |
| `uca_annotations/` | `UCFCrime_Train/Val/Test.json` from github.com/Xuange923/Surveillance-Video-Understanding |

## Step 1: upload (WinSCP)

Left panel: `C:\Users\LENOVO\Desktop\Vlil40\Current work\`
Right panel: `/home/y_rajeshwar/vlil_40/`
Drag the folder `v6_crime_moments` across. You should get `/home/y_rajeshwar/vlil_40/v6_crime_moments/`.

## Step 2: fix Windows line endings (PuTTY)

```bash
cd ~/vlil_40/v6_crime_moments
sed -i 's/\r$//' *.py *.sh
```

## Step 3: submit prep -> seed 42 -> seed 7 -> seed 123 as one chain

```bash
cd ~/vlil_40/v6_crime_moments
PREP=$(qsub run_prepare_v6.sh); echo $PREP
S1=$(qsub -W depend=afterok:$PREP -v SEED=42 run_train_v6.sh); echo $S1
S2=$(qsub -W depend=afterany:$S1 -v SEED=7 run_train_v6.sh); echo $S2
S3=$(qsub -W depend=afterany:$S2 -v SEED=123 run_train_v6.sh); echo $S3
qstat -u y_rajeshwar
```

Same PuTTY window for all lines. Waiting jobs show **H** (the chain), not the old ngpus problem.

## Step 4: watch live (Ctrl+C stops watching, not the job)

```bash
tail -f ~/vlil_40/v6_crime_moments/logs/prep_live.log
cat ~/vlil_40/v6_crime_moments/logs/prep_report.txt
tail -f ~/vlil_40/v6_crime_moments/runs/seed42_ucf101/logs/train_live.log
grep "Epoch" ~/vlil_40/v6_crime_moments/runs/seed42_ucf101/logs/train_live.log
```

## Step 5: results

```bash
cat ~/vlil_40/v6_crime_moments/results_all_runs.csv
tail -n 60 ~/vlil_40/v6_crime_moments/runs/seed123_ucf101/logs/train_live.log
```

The end of each log prints the mean over the seeds finished so far.

## If a job is killed

Submit the same `qsub -v SEED=.. run_train_v6.sh` again. It resumes from `runs/seed.._ucf101/checkpoints/last.pth`.

## Spot-check the training windows

`splits/crime_windows.csv` lists every window with the UCA sentences that produced it. Download it with
WinSCP and open it in Excel to check a few per class.
