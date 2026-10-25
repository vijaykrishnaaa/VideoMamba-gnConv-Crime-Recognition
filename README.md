# VideoMamba + gnConv2D for crime-type recognition

Recognises which crime a surveillance clip shows (13 crime types + Normal) with
**VideoMamba-Small** extended by **HorNet's recursive gated convolution (gnConv2D)** in its
last 8 layers. It is the perception stage of a first-response robot; the robot side
(quadruped SLAM and exploration) is in
[Autonomous-Quadruped-SLAM-Navigation-Perception](https://github.com/yokshith09/Autonomous-Quadruped-SLAM-Navigation-Perception).

## Results

| Experiment | Data and protocol | Result |
|---|---|---|
| V5 | UCF-Crime, official 4-fold recognition split, 14 classes | **37.05 %** mean test accuracy (folds 41.07 / 29.76 / 36.90 / 40.48) |
| V5 | same models, 119 test videos scored only inside the annotated crime window | 42.86 % (whole video: 36.97 %) |
| V6 | UCF-Crime crime moments (UCA windows for training, official windows for test), 13 classes, 3 seeds | 39.52 % accuracy, 32.83 % balanced accuracy |
| V4 | own 13-class violence dataset (not public) | 73.47 % (70.45 % without 15 test clips that share a recording with training clips) |

Published results on the same UCF-Crime recognition protocol: C3D 23.0 %, TCNN 28.4 %
(Sultani et al., CVPR 2018), TCNN + motion-aware features 31.0 % (Zhu and Newsam, BMVC 2019).

Model: 116.2M parameters (99.2M trained), 370 GFLOPs per 8-frame clip, 59 ms per clip on a
Tesla V100 (batch 1, float32).

## Model

- 8 frames of 224 x 224 -> Conv3d patch embedding -> 1,569 tokens of 384 -> 24 bidirectional Mamba layers -> CLS token -> linear classifier.
- Layers 0-15 frozen (Kinetics-400 -> UCF-101 weights); layers 16-23 trained.
- In each trained layer, gnConv2D (order 5, GhostModule input, kernels 3, 5, 7, 1, zero-initialised output) runs beside the Conv1d of both scan directions, on each frame's 14 x 14 patch grid.
- `mamba_simple_v7.py` is the Mamba layer with gnConv2D. It fixes how frames are reshaped for gnConv2D and where the CLS token is skipped in the backward scan, and builds no gnConv2D in frozen layers (there it always output 0).

## Repository layout

| Folder | What it is |
|---|---|
| `v1_ucf101/` | Kinetics-400 -> UCF-101 fine-tuning (the transfer step, 94.6 %) |
| `v1_crime/`, `v3_crime_transfer/` | early runs on our own dataset (V1, V3; superseded) |
| `v4_crime_fixed/` | V4: fixed backbone, own dataset; logs, splits and results |
| `v5_ucf_crime/` | V5: UCF-Crime official 4-fold recognition; `runs/foldK_ucf101/` hold logs and results |
| `v6_crime_moments/` | V6: training on crime moments from the UCA annotations |
| `environment/` | `environment.yml` and `pip_freeze.txt` of the conda env used |

Each experiment has `prepare_*.py` (decodes videos once into a frame cache and writes the splits),
`train_*.py` (training, validation-based model selection, one final test) and PBS job scripts
`run_*.sh`. `V5_RUN_STEPS.md` / `V6_RUN_STEPS.md` give the exact commands.

## Requirements

- Python 3.10, PyTorch 2.1.1 + CUDA 11.8, a CUDA GPU (Mamba's selective scan has no CPU path).
- `mamba_ssm` 1.1.1 and `causal_conv1d` built from [hustvl/Vim](https://github.com/hustvl/Vim).
- The [VideoMamba](https://github.com/OpenGVLab/VideoMamba) repository cloned to `~/vlil_40/VideoMamba` (its `videomamba/video_sm/models/videomamba.py` builds the model).
- UCF-Crime from the [dataset page](https://www.crcv.ucf.edu/projects/real-world/); for V6 also the
  [UCA annotations](https://github.com/Xuange923/Surveillance-Video-Understanding) in `v6_crime_moments/uca_annotations/`.

Paths are set for the HPC account the project ran on (`~/vlil_40`, `~/final_year_project_29/...`); change the
constants at the top of each script for another machine. Checkpoints and frame caches are not included.

## Credits

VideoMamba (Li et al., 2024), HorNet (Rao et al., 2022), Mamba (Gu and Dao, 2023), GhostNet (Han et al., 2020),
UCF-Crime (Sultani et al., 2018), UCA (Yuan et al., 2024).
