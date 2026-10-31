#!/usr/bin/env python
"""
Fine-tune the pretrained DiffWater checkpoint on EUVP with the ORIGINAL noise schedule (Run A)
or the MODIFIED schedule theta_t ** X (Run B).  Group 25, Noise Schedule Ablation.

Where this file sits: it is an ADDED file that imports the unmodified DiffWater code
(Guan-MS/DiffWater, nested folder DiffWater/DiffWater) and replaces exactly ONE method at run time:
GaussianDiffusion.set_new_noise_schedule  (model/ddpm_modules/diffusion.py).  The replacement is
copied below with every changed line marked "# MODIFIED".  Everything else in this file is "# ADDED".

Definition used here:  theta_t = alpha_bar_t = prod_{s<=t} (1 - beta_s).
  Run A: theta_t          (exponent 1, the repo's own schedule, code path untouched)
  Run B: theta_t ** X     (X below).  The per-step alpha_t / beta_t, the posterior variance and the
         posterior mean coefficients are all re-derived from the new cumulative product, so training
         (q_sample) and sampling (p_sample) use one consistent schedule.

Typical use (see the Colab cells):
  python finetune.py --benchmark   --repo_dir ... --ckpt_prefix ...
  python finetune.py --smoke       --run B ...            # 2-minute code-path test
  python finetune.py --run A       --repo_dir ... --ckpt_prefix ... --train_root ... --eval_root ... --out_dir ...
  python finetune.py --run B       (same arguments; only --run differs)
  python finetune.py --run 0       --repo_dir ... --ckpt_prefix ... --eval_root ... --out_dir ...
      Run 0 = the pretrained checkpoint, no training, sampled by exactly the same code and per-image
      seeds as the final evaluation of A and B (original schedule).  --train_root is not needed.

Hyperparameters are identical for A and B; only the exponent differs.  Everything is seeded
(per-epoch seeds for data order / timesteps / noise, per-image seeds for sampling).
The job can be stopped and restarted with the same command: it resumes from out_dir/resume.pt
and skips images that already exist.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
import types
from functools import partial

import numpy as np
import torch
import yaml

# ============================ ADDED: the one experimental knob ============================
X = 2.00   # exponent in theta_t ** X for the MODIFIED schedule (Run B).  Run A always uses exponent 1.
# ==========================================================================================

# ADDED: fixed settings shared by Run A and Run B (do not change between the two runs)
DEFAULT_SEED = 1234
DEFAULT_LR = 1e-5          # the authors pre-trained with 3e-6 at batch size 1; short fine-tune uses 1e-5
DEFAULT_EPOCHS = 10        # must be a multiple of 5 (samples are written at 5 equally spaced epochs)
DEFAULT_BATCH = 4
DEFAULT_N_SAMPLES = 3      # images sampled at each of the 5 epochs (>= 3 required by the assignment)

EXPONENT = 1.0             # ADDED: active exponent, set in main() (A -> 1.0, B -> X)
R = types.SimpleNamespace()  # ADDED: holds the DiffWater modules once the repo path is on sys.path
LOG_PATH = None


# ------------------------------------------------------------------------------------------
# ADDED: logging helper (prints and appends to out_dir/run.log so progress survives a disconnect)
# ------------------------------------------------------------------------------------------
def log(msg):
    line = "{} {}".format(time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------------------------------
# MODIFIED: copy of GaussianDiffusion.set_new_noise_schedule from DiffWater
#           (model/ddpm_modules/diffusion.py).  Lines marked "# MODIFIED" are the only changes.
# ------------------------------------------------------------------------------------------
def set_new_noise_schedule_mod(self, schedule_opt, device):
    to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
    betas = R.D.make_beta_schedule(
        schedule=schedule_opt['schedule'],
        n_timestep=schedule_opt['n_timestep'],
        linear_start=schedule_opt['linear_start'],
        linear_end=schedule_opt['linear_end'])
    betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    if EXPONENT != 1.0:                                           # MODIFIED
        alphas_cumprod = alphas_cumprod ** EXPONENT               # MODIFIED  theta_t -> theta_t ** x
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
    if EXPONENT != 1.0:                                           # MODIFIED
        alphas = alphas_cumprod / alphas_cumprod_prev             # MODIFIED  per-step alpha_t implied by new product
        betas = 1. - alphas                                       # MODIFIED  per-step beta_t implied by new product

    timesteps, = betas.shape
    self.num_timesteps = int(timesteps)
    self.register_buffer('betas', to_torch(betas))
    self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
    self.register_buffer('alphas_cumprod_prev',
                         to_torch(alphas_cumprod_prev))

    # calculations for diffusion q(x_t | x_{t-1}) and others
    self.register_buffer('sqrt_alphas_cumprod',
                         to_torch(np.sqrt(alphas_cumprod)))
    self.register_buffer('sqrt_one_minus_alphas_cumprod',
                         to_torch(np.sqrt(1. - alphas_cumprod)))
    self.register_buffer('log_one_minus_alphas_cumprod',
                         to_torch(np.log(1. - alphas_cumprod)))
    self.register_buffer('sqrt_recip_alphas_cumprod',
                         to_torch(np.sqrt(1. / alphas_cumprod)))
    self.register_buffer('sqrt_recipm1_alphas_cumprod',
                         to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * \
        (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    self.register_buffer('posterior_variance',
                         to_torch(posterior_variance))
    self.register_buffer('posterior_log_variance_clipped', to_torch(
        np.log(np.maximum(posterior_variance, 1e-20))))
    self.register_buffer('posterior_mean_coef1', to_torch(
        betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
    self.register_buffer('posterior_mean_coef2', to_torch(
        (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


# ------------------------------------------------------------------------------------------
# ADDED: model construction.  The schedule buffers (alphas_cumprod, betas, ...) are registered
# buffers, so they are stored INSIDE the checkpoint's state_dict.  Loading the pretrained weights
# therefore overwrites a modified schedule with the original one.  We re-apply the schedule after
# every load and assert it, otherwise Run B would silently be Run A.
# ------------------------------------------------------------------------------------------
def load_config(args):
    cfg = yaml.safe_load(open(os.path.join(args.repo_dir, "config", "config.yml")))
    cfg["gpu_ids"] = [0]
    cfg["phase"] = "val"                       # val phase: loads <prefix>_gen.pth only, no optimizer, no step counters
    cfg["path"]["resume_state"] = args.ckpt_prefix
    return cfg


def apply_and_check_schedule(diffusion, cfg):
    sched = cfg["model"]["beta_schedule"]["train"]
    diffusion.netG.set_new_noise_schedule(sched, diffusion.device)    # bypasses DDPM's phase guard on purpose
    betas = np.asarray(R.D.make_beta_schedule(
        schedule=sched["schedule"], n_timestep=sched["n_timestep"],
        linear_start=sched["linear_start"], linear_end=sched["linear_end"]), dtype=np.float64)
    expected = np.cumprod(1.0 - betas) ** EXPONENT
    got = diffusion.netG.alphas_cumprod.double().cpu().numpy()
    assert np.allclose(got, expected, rtol=1e-5, atol=0), "schedule buffers do not match exponent %s" % EXPONENT
    assert ((got > 0) & (got < 1)).all() and (np.diff(got) < 0).all(), "alpha_bar must lie in (0,1) and decrease"
    b = diffusion.netG.betas.double().cpu().numpy()
    assert (b > 0).all() and (b < 1).all(), "per-step beta must lie in (0,1)"
    return dict(exponent=EXPONENT, abar_first=float(got[0]), abar_last=float(got[-1]),
                beta_min=float(b.min()), beta_max=float(b.max()), T=int(len(got)))


def build_model(args):
    cfg = load_config(args)
    opt = R.Logger.dict_to_nonedict(cfg)
    diffusion = R.Model.create_model(opt)      # loads the pretrained generator (strict), incl. original schedule buffers
    return diffusion, cfg


# ------------------------------------------------------------------------------------------
# ADDED: sampling with fixed per-image seed and recorded trajectory
# ------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_with_traj(netG, cond, seed):
    """Same reverse process as GaussianDiffusion.p_sample_loop (all T steps, clip_denoised=True),
    but with a fixed seed and the image x_t recorded at 4 equally spaced t plus the final x_0."""
    T = netG.num_timesteps
    traj_ts = [int(round(v)) for v in np.linspace(T - 1, 0, 5)][:4]     # T=2000 -> 1999, 1499, 1000, 500
    torch.manual_seed(seed)
    img = torch.randn(cond.shape, device=cond.device)                    # x_{T-1}
    b = cond.shape[0]
    frames = []
    for i in reversed(range(T)):
        if i in traj_ts:
            frames.append(img.detach().float().cpu())                    # x_i before the step at i
        img = netG.p_sample(img, torch.full((b,), i, device=cond.device, dtype=torch.long), condition_x=cond)
    frames.append(img.detach().float().cpu())                            # final x_0
    return img.detach().float().cpu(), frames, traj_ts


def to_u8(t):
    return R.Metrics.tensor2img(t.squeeze(0))


def sample_and_save(netG, item, k, out_dir, seed):
    """k is the 1-based image index (same order as Run 0). Skips if already done."""
    paths = {n: os.path.join(out_dir, "idx{:02d}_{}.png".format(k, n)) for n in ("input", "output", "target", "traj")}
    if all(os.path.exists(p) for p in paths.values()):
        return False
    os.makedirs(out_dir, exist_ok=True)
    cond = item["input"].unsqueeze(0).to("cuda")
    netG.eval()
    out, frames, _ = sample_with_traj(netG, cond, seed)
    netG.train()
    R.Metrics.save_img(to_u8(item["input"].unsqueeze(0)), paths["input"])
    R.Metrics.save_img(to_u8(item["target"].unsqueeze(0)), paths["target"])
    R.Metrics.save_img(to_u8(out), paths["output"])
    R.Metrics.save_img(np.concatenate([to_u8(f) for f in frames], axis=1), paths["traj"])
    return True


def metrics_rows(dir_, idxs):
    from PIL import Image
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    load = lambda k, n: np.array(Image.open(os.path.join(dir_, "idx{:02d}_{}.png".format(k, n))).convert("RGB"))
    rows = []
    for k in idxs:
        out, tgt, inp = load(k, "output"), load(k, "target"), load(k, "input")
        rows.append(dict(idx=k,
                         psnr_out=psnr(tgt, out, data_range=255), ssim_out=ssim(tgt, out, channel_axis=2, data_range=255),
                         psnr_in=psnr(tgt, inp, data_range=255), ssim_in=ssim(tgt, inp, channel_axis=2, data_range=255)))
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------------
# ADDED: benchmark (random data, no dataset needed): seconds per training step and memory
# ------------------------------------------------------------------------------------------
def run_benchmark(args):
    diffusion, cfg = build_model(args)
    info = apply_and_check_schedule(diffusion, cfg)
    log("schedule OK: %s" % info)
    netG = diffusion.netG
    netG.train()
    optim = torch.optim.Adam(netG.parameters(), lr=args.lr)
    for bs in (1, 2, 4, 8):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            data = dict(input=torch.randn(bs, 3, 256, 256, device="cuda").clamp(-1, 1),
                        target=torch.randn(bs, 3, 256, 256, device="cuda").clamp(-1, 1))
            for it in range(8):
                if it == 3:
                    torch.cuda.synchronize()
                    t0 = time.time()
                optim.zero_grad(set_to_none=True)
                loss = netG(data).sum() / (bs * 3 * 256 * 256)
                loss.backward()
                optim.step()
            torch.cuda.synchronize()
            per = (time.time() - t0) / 5
            log("batch %d: %.2f s/step  (%.2f s/image, %.1f img/s)  peak memory %.1f GB"
                % (bs, per, per / bs, bs / per, torch.cuda.max_memory_allocated() / 1e9))
        except torch.cuda.OutOfMemoryError:
            log("batch %d: out of memory" % bs)
            optim.zero_grad(set_to_none=True)
            break


# ------------------------------------------------------------------------------------------
# ADDED: training / evaluation driver
# ------------------------------------------------------------------------------------------
def main():
    global EXPONENT, LOG_PATH
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=["0", "A", "B"], help="0 = pretrained, no fine-tuning; A = original schedule; B = theta_t ** X")
    p.add_argument("--repo_dir", required=True, help="the nested DiffWater folder (contains test.py, model/, data/)")
    p.add_argument("--ckpt_prefix", required=True, help="path of the pretrained checkpoint WITHOUT the _gen.pth suffix")
    p.add_argument("--train_root", help="folder with input_256/ and target_256/ (training pairs)")
    p.add_argument("--eval_root", help="folder with input_256/ and target_256/ (the 30 Run-0 test pairs)")
    p.add_argument("--out_dir", help="output folder (put it on Drive)")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--n_eval", type=int, default=30, help="number of test images for the final evaluation (first N of the 30)")
    p.add_argument("--stage", choices=["all", "train", "eval"], default="all")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny code-path test (T=20, 2 steps per epoch)")
    args = p.parse_args()

    sys.path.insert(0, args.repo_dir)
    os.chdir(args.repo_dir)
    import core.logger as Logger
    import core.metrics as Metrics
    import data as Data
    import model as Model
    from model.ddpm_modules import diffusion as D
    R.Logger, R.Metrics, R.Data, R.Model, R.D = Logger, Metrics, Data, Model, D
    D.GaussianDiffusion.set_new_noise_schedule = set_new_noise_schedule_mod      # MODIFIED (monkeypatch)

    EXPONENT = X if args.run == "B" else 1.0           # Run 0 and Run A use the original schedule
    if args.benchmark:
        run_benchmark(args)
        return
    assert args.run and args.eval_root and args.out_dir, "need --run --eval_root --out_dir"
    assert args.run == "0" or args.train_root, "--train_root is required for Run A and Run B"
    if args.run == "0":
        args.stage = "eval"                                  # Run 0: no training, evaluation of the pretrained weights only
    assert args.epochs % 5 == 0, "--epochs must be a multiple of 5"
    if args.smoke:
        args.out_dir = args.out_dir.rstrip("/") + "_smoke"
        args.n_eval, args.n_samples = 2, 1
    os.makedirs(args.out_dir, exist_ok=True)
    LOG_PATH = os.path.join(args.out_dir, "run.log")
    log("run=%s exponent=%s args=%s" % (args.run, EXPONENT, vars(args)))

    from torch.utils.data import DataLoader

    train_set = (Data.create_dataset(dict(dataroot=args.train_root, resolution=256, data_len=-1, name="EUVP-train"), "train")
                 if args.run != "0" else [])
    eval_set = Data.create_dataset(dict(dataroot=args.eval_root, resolution=256, data_len=-1, name="EUVP-test"), "val")
    n_eval = min(args.n_eval, len(eval_set))
    log("train pairs: %d   eval pairs available: %d (using %d)" % (len(train_set), len(eval_set), n_eval))

    diffusion, cfg = build_model(args)
    if args.smoke:
        # tiny T for a code-path test. Set AFTER the checkpoint is loaded: the checkpoint stores the schedule
        # buffers with T=2000, so building the model with T=20 would fail the strict load (size mismatch).
        for ph in ("train", "val"):
            cfg["model"]["beta_schedule"][ph]["n_timestep"] = 20
    netG = diffusion.netG
    optim = torch.optim.Adam(netG.parameters(), lr=args.lr)     # fresh optimizer (the authors' _opt.pth also resets step/epoch counters)

    sample_epochs = [args.epochs // 5 * j for j in range(1, 6)]
    log("sample epochs: %s   n_samples per epoch: %d" % (sample_epochs, args.n_samples))
    state_path = os.path.join(args.out_dir, "resume.pt")
    csv_path = os.path.join(args.out_dir, "train_log.csv")
    step_csv = os.path.join(args.out_dir, "train_steps.csv")

    start_epoch, rows = 0, []
    if args.run != "0" and not args.smoke and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)     # our own file
        assert st["run"] == args.run and abs(st["exponent"] - EXPONENT) < 1e-12, "resume.pt belongs to a different run"
        netG.load_state_dict(st["gen"])
        optim.load_state_dict(st["opt"])
        start_epoch = st["epoch"]
        if os.path.exists(csv_path):
            rows = [r for r in csv.DictReader(open(csv_path)) if int(r["epoch"]) <= start_epoch]
            write_csv(csv_path, rows, ["epoch", "train_loss"])
        if os.path.exists(step_csv):
            kept = [r for r in csv.reader(open(step_csv)) if int(r[0]) <= start_epoch]
            with open(step_csv, "w", newline="") as f:
                csv.writer(f).writerows(kept)
        log("resumed from epoch %d" % start_epoch)
        del st
    if args.stage == "eval" and args.run != "0":
        assert start_epoch == args.epochs, "--stage eval needs a finished training run (resume.pt at epoch %d)" % args.epochs
    info = apply_and_check_schedule(diffusion, cfg)             # after every load
    log("schedule OK: %s" % info)
    json.dump(dict(run=args.run, exponent=EXPONENT, schedule=info, args=vars(args)),
              open(os.path.join(args.out_dir, "run_config.json"), "w"), indent=2)

    def post_epoch(e):
        if e not in sample_epochs:
            return
        d = os.path.join(args.out_dir, "samples", "epoch_%02d" % e)
        t0 = time.time()
        for k in range(1, args.n_samples + 1):
            if sample_and_save(netG, eval_set[k - 1], k, d, seed=args.seed * 1000 + k):
                log("epoch %d sample %d/%d saved (%.0f s total)" % (e, k, args.n_samples, time.time() - t0))
        r = metrics_rows(d, range(1, args.n_samples + 1))
        write_csv(os.path.join(d, "metrics.csv"), r, list(r[0].keys()))
        log("epoch %d sample PSNR %.2f SSIM %.3f" % (e, np.mean([x["psnr_out"] for x in r]), np.mean([x["ssim_out"] for x in r])))

    def save_state(e):
        tmp = state_path + ".tmp"
        torch.save(dict(gen=netG.state_dict(), opt=optim.state_dict(), epoch=e, run=args.run, exponent=EXPONENT), tmp)
        os.replace(tmp, state_path)
        log("saved resume state at epoch %d" % e)

    # ------------------------------- training -------------------------------
    if args.stage in ("all", "train"):
        if start_epoch > 0:
            post_epoch(start_epoch)                             # finish samples interrupted by a disconnect
        t_train = time.time()
        for e in range(start_epoch + 1, args.epochs + 1):
            torch.manual_seed(args.seed + e)                    # same data order / timesteps / noise in A and B
            g = torch.Generator()
            g.manual_seed(args.seed + e)
            loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                num_workers=2, pin_memory=True, generator=g)
            netG.train()
            losses, t0 = [], time.time()
            for s, batch in enumerate(loader, start=1):
                if args.smoke and s > 2:
                    break
                diffusion.feed_data(batch)
                optim.zero_grad(set_to_none=True)
                b, c, h, w = diffusion.data["target"].shape
                loss = netG(diffusion.data).sum() / int(b * c * h * w)     # same normalisation as DDPM.optimize_parameters
                loss.backward()
                optim.step()
                losses.append(loss.item())
                if s % 50 == 0:
                    log("epoch %d step %d/%d loss %.4f  (%.2f s/step)" % (e, s, len(loader), np.mean(losses[-50:]), (time.time() - t0) / s))
            rows.append(dict(epoch=e, train_loss=float(np.mean(losses))))
            write_csv(csv_path, rows, ["epoch", "train_loss"])
            with open(step_csv, "a", newline="") as f:
                csv.writer(f).writerows([[e, i + 1, l] for i, l in enumerate(losses)])
            log("epoch %d done: train_loss %.5f  (%.1f min, %.1f min since start)"
                % (e, rows[-1]["train_loss"], (time.time() - t0) / 60, (time.time() - t_train) / 60))
            if e in sample_epochs and (not args.smoke or e == args.epochs):
                save_state(e)                                   # save BEFORE sampling so samples match this state on resume
            post_epoch(e)

    # ------------------------------ final evaluation ------------------------------
    if args.stage in ("all", "eval"):
        fin = os.path.join(args.out_dir, "final_eval")
        last_dir = os.path.join(args.out_dir, "samples", "epoch_%02d" % args.epochs)
        t0 = time.time()
        for k in range(1, n_eval + 1):
            reuse = [os.path.join(last_dir, "idx{:02d}_{}.png".format(k, n)) for n in ("input", "output", "target", "traj")]
            if args.stage == "all" and all(os.path.exists(p) for p in reuse):
                os.makedirs(fin, exist_ok=True)                 # same weights, same seed: reuse the last-epoch sample
                for p in reuse:
                    q = os.path.join(fin, os.path.basename(p))
                    if not os.path.exists(q):
                        shutil.copy(p, q)
                continue
            if sample_and_save(netG, eval_set[k - 1], k, fin, seed=args.seed * 1000 + k):
                log("final eval %d/%d done (%.1f min elapsed)" % (k, n_eval, (time.time() - t0) / 60))
        r = metrics_rows(fin, range(1, n_eval + 1))
        write_csv(os.path.join(args.out_dir, "eval_metrics.csv"), r, list(r[0].keys()))
        log("FINAL  PSNR %.3f  SSIM %.4f  over %d images  (input vs ref: PSNR %.3f SSIM %.4f)"
            % (np.mean([x["psnr_out"] for x in r]), np.mean([x["ssim_out"] for x in r]), len(r),
               np.mean([x["psnr_in"] for x in r]), np.mean([x["ssim_in"] for x in r])))
        if args.run != "0":
            torch.save(netG.state_dict(), os.path.join(args.out_dir, "final_gen.pth"))
    log("ALL DONE")


if __name__ == "__main__":
    main()
