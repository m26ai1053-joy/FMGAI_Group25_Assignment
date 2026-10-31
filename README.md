# Group 25: Noise Schedule Ablation in Diffusion Models (DiffWater, EUVP)

Foundation Models and Generative AI. Group 25: Sameer Sarkar (M26AI1107), Joydeep Roy (M26AI1053), Krishnendu Halder (M26AI1054), Samanwita Patra (M26AI1106).

**The model is DiffWater (Guan-MS/DiffWater), not UW-DiffPhys.** Our Phase 1 declaration named UW-DiffPhys; why we changed is explained in the report (Part A, item 2). Task, dataset (EUVP) and exponent x = 2.00 are as declared; the image size is 256 x 256.

## What is in this folder

| Folder | Content |
|---|---|
| `report/` | `Group25_Final_Report.pdf` (and `.docx`): revised declaration and the report with the nine required sections |
| `code/finetune.py` | **The fine-tuning code.** `X = 2.00` is at line 46. The five schedule lines 86, 87, 89, 90, 91, the header comment (line 73) and the line that installs the method (line 289) are marked `# MODIFIED`; each other block starts with `# ADDED` and the header says everything else is added |
| `code/val_loss.py` | post-hoc validation (200 held-out pairs); imports `finetune.py` unchanged |
| `code/notebooks/RunAB_Finetune.ipynb` | Colab notebook that builds the data and launches Run 0, Run A or Run B (set `RUN`) |
| `code/notebooks/PostHoc_Validation.ipynb` | Colab notebook for the post-hoc validation (one cell for Run A, one for Run B) |
| `code/DiffWater/` | the DiffWater code at commit `7db68f34060bbf1d7c5b8153664fb16a3dcda1ae`, **unmodified** (exported from the authors' repository, Apache-2.0, licence included). `finetune.py` imports it and replaces one method at run time |
| `logs/` | fine-tuning logs: `RunA/train_log.csv`, `RunB/train_log.csv` (columns `epoch,train_loss`), step losses, `run.log`, `run_config.json`, per-image test scores (`eval_metrics.csv`) for Run 0, A and B, the test split, the post-hoc validation outputs, and the console outputs of the Colab cells |
| `samples/` | generated samples at epochs 2, 4, 6, 8, 10 for test images 1 to 3: `RunA/epoch_NN/`, `RunB/epoch_NN/` (files `idxKK_input|output|target|traj.png`, `metrics.csv`); `overview_samples_Run*.jpg` are labelled contact sheets; `Run0/` has the Run 0 outputs for the same three images |
| `trajectories/` | inference trajectories of the final models: the noisy sample x_t at t = 1999, 1499, 1000, 500 and the final x_0, for test images 1 to 3, one file per timestep (`RunA_image01_t1999.png` ...) and one labelled strip per image |
| `figures/` | the figures used in the report |
| `checkpoints/` | final weights of Run A and Run B, and the pretrained starting checkpoint (`original/`); see `checkpoints/README.txt` |

## Environment and seed

Google Colab, Tesla T4 GPU, Python 3.13.16, PyTorch 2.11.0+cu130, DiffWater commit `7db68f34060bbf1d7c5b8153664fb16a3dcda1ae`, `finetune.py` md5 `38c97203e8ccafdfb80cb2be74b0bb3a`. The notebook installs `tensorboardX scikit-image lmdb pyyaml kagglehub pandas` with pip. **Random seed 1234** for everything (epoch e uses seed 1234 + e; test image k is sampled with seed 1234 x 1000 + k).

## How to reproduce

1. Open `code/notebooks/RunAB_Finetune.ipynb` in Colab (T4 GPU). Put `finetune.py` in `/content` and the pretrained checkpoint `I390000_E78_gen.pth` (the DiffWater authors' file, a copy is in `checkpoints/original/`) in `My Drive/DW/LSUI/`. The setup cell clones DiffWater at the pinned commit (the same code as in `code/DiffWater/`).
2. The data cell downloads EUVP from Kaggle and builds the 30 test pairs (split fingerprint `f8cc4714a30c`) and the 800 training pairs (fingerprint `6a9f33de995c`). Both must print exactly these values.
3. The launch cell runs, with `RUN` = `A`, `B` or `0`, the commands below (it prints the command it runs and checks the md5 of `finetune.py`, the commit and the fingerprints first):

```bash
# Run A: original schedule
python -u /content/finetune.py --run A --repo_dir /content/DiffWater/DiffWater \
  --ckpt_prefix /content/drive/MyDrive/DW/LSUI/I390000_E78 \
  --train_root /content/euvp_train --eval_root /content/euvp_test \
  --out_dir /content/drive/MyDrive/FoundationModel-Assignment/runs/RunA \
  --epochs 10 --batch_size 4 --n_eval 30 --n_samples 3

# Run B: schedule raised to the power X = 2.00 (same command, only --run and --out_dir differ)
python -u /content/finetune.py --run B --repo_dir /content/DiffWater/DiffWater \
  --ckpt_prefix /content/drive/MyDrive/DW/LSUI/I390000_E78 \
  --train_root /content/euvp_train --eval_root /content/euvp_test \
  --out_dir /content/drive/MyDrive/FoundationModel-Assignment/runs/RunB \
  --epochs 10 --batch_size 4 --n_eval 30 --n_samples 3

# Run 0: pretrained model, no fine-tuning (no --train_root)
python -u /content/finetune.py --run 0 --repo_dir /content/DiffWater/DiffWater \
  --ckpt_prefix /content/drive/MyDrive/DW/LSUI/I390000_E78 \
  --eval_root /content/euvp_test \
  --out_dir /content/drive/MyDrive/FoundationModel-Assignment/runs/Run0 \
  --epochs 10 --batch_size 4 --n_eval 30 --n_samples 3
```

4. A healthy run prints `schedule OK` first: `exponent 1.0, abar_last 4.386e-05, beta_max 0.01` for Run 0 and Run A, `exponent 2.0, abar_last 1.924e-09, beta_max 0.0199` for Run B. Run A and Run B take about 4.5 hours each on a T4, Run 0 about 2.7 hours.
5. If Colab disconnects, run the setup, data and launch cells again in the same account: the job resumes from `resume.pt` and skips finished images.
6. Post-hoc validation: `code/notebooks/PostHoc_Validation.ipynb` runs `code/val_loss.py` for each run (needs that run's `final_gen.pth` in `My Drive/FoundationModel-Assignment/runs/RunA|RunB/`). Expected validation fingerprint `24416ebc2732`.

## Results (30 held-out Underwater Dark test pairs, output against reference)

| Condition | PSNR (dB) | SSIM |
|---|---|---|
| Input (colour-compensated), no model | 20.647 | 0.8887 |
| Run 0 (pretrained) | 16.908 | 0.7925 |
| Run A (fine-tuned, original schedule) | 14.360 | 0.7701 |
| Run B (fine-tuned, theta^2, x = 2.00) | 12.849 | 0.7631 |

Run A is higher than Run B by 1.51 dB PSNR (standard error 0.40); SSIM does not separate them. Details, figures and the analysis are in the report.

## Model Checkpoints
[Google Drive Link](https://drive.google.com/drive/folders/1R1bP67GS5IFrkBvyigJAy6ZWeGnSh4Uk?usp=sharing)

| Model Type | Drive Name |
|------------|-----------|
| Run 0 (pretrained)                    | original |
| Run A (fine-tuned, original schedule) | RunA     |
| Run B (fine-tuned, theta^2, x = 2.00)  | RunB     |
