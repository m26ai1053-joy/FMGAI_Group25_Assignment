#!/usr/bin/env python
"""
POST-HOC validation loss for Run A or Run B.  Group 25, Noise Schedule Ablation.   (ADDED file)

What it does
  Scores the pretrained weights and the fine-tuned weights (final_gen.pth of the run) on held-out
  validation pairs that were NOT used for training and are NOT among the 30 test pairs, using the
  training objective itself: the L1 error between the true noise and the noise the network predicts.
  This is "post hoc": it runs after fine-tuning is finished, and nothing in Run A or Run B was
  chosen from it.  It does not replace validation during fine-tuning.

How it stays comparable with training
  * It imports finetune.py (unchanged) and uses ITS schedule code, so Run B is evaluated under
    theta_t ** X exactly as it was trained, and the schedule is asserted after every load.
  * The loss is the same quantity as the training loss: L1(noise, predicted noise), summed and divided
    by (batch * 3 * 256 * 256), i.e. a mean absolute error per element.
  * Training draws one random timestep per image.  Here every image is scored at N_T fixed timesteps,
    the midpoints of N_T equal bins of 0..T-1 (T=2000, N_T=10: 100, 300, ..., 1900), and the average over
    them estimates the same expectation without sampling noise.  The noise tensor for a given
    (batch, timestep) is fixed by a seed, so the pretrained and fine-tuned models see identical
    inputs and the comparison is paired per image.
  * Dropout is 0 in DiffWater's config, so eval() and train() mode give the same loss.

Usage (see the notebook):
  python val_loss.py --run A --repo_dir /content/DiffWater/DiffWater --ckpt_prefix <pretrained, no _gen.pth>
         --final_ckpt <.../RunA/final_gen.pth> --val_root /content/euvp_val --out_dir <folder>
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=["A", "B"], required=True, help="A: original schedule; B: theta_t ** X (X from finetune.py)")
    p.add_argument("--repo_dir", required=True, help="the nested DiffWater folder")
    p.add_argument("--ckpt_prefix", required=True, help="pretrained checkpoint path WITHOUT the _gen.pth suffix")
    p.add_argument("--final_ckpt", required=True, help="final_gen.pth written by finetune.py for this run")
    p.add_argument("--val_root", required=True, help="folder with input_256/ and target_256/ (validation pairs)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--finetune_py", default="/content/finetune.py")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_t", type=int, default=10, help="number of fixed timesteps per image")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit", type=int, default=0, help="score only the first N validation pairs (dry run); 0 = all")
    p.add_argument("--n_train_path", type=int, default=3, help="repeats of the random-timestep check that mirrors training")
    return p.parse_args()


@torch.no_grad()
def evaluate(netG, inp, tgt, ts, bs, seed0, label, F, dev="cuda"):
    """Per-image L1 noise-prediction error at each fixed timestep.  Returns array (N, len(ts))."""
    netG.eval()
    n = inp.shape[0]
    per = np.zeros((n, len(ts)))
    t0 = time.time()
    for bi, s in enumerate(range(0, n, bs)):
        x_in = inp[s:s + bs].to(dev)
        x0 = tgt[s:s + bs].to(dev)
        b, c, h, w = x0.shape
        for k, t_int in enumerate(ts):
            g = torch.Generator(device=dev)
            g.manual_seed(seed0 + bi * 100 + k)                      # same noise for every model and every run
            noise = torch.randn(x0.shape, generator=g, device=dev)
            t = torch.full((b,), t_int, device=dev, dtype=torch.long)
            x_noisy = netG.q_sample(x_start=x0, t=t, noise=noise)
            recon = netG.denoise_fn(torch.cat([x_in, x_noisy], dim=1), t)       # same call as GaussianDiffusion.p_losses
            per[s:s + b, k] = (noise - recon).abs().mean(dim=(1, 2, 3)).cpu().numpy()
            if bi == 0 and k == 0:                                   # my loss must equal the repository's loss function
                ref = netG.loss_func(noise, recon).item() / (b * c * h * w)
                assert abs(ref - per[s:s + b, k].mean()) < 1e-4 * max(1.0, abs(ref)), "loss definition differs from p_losses"
        if bi % 10 == 9 or s + bs >= n:
            F.log("%s: %d/%d images scored (%.1f min)" % (label, min(s + bs, n), n, (time.time() - t0) / 60))
    return per


@torch.no_grad()
def train_path_estimate(netG, inp, tgt, bs, seed, dev="cuda"):
    """The training forward pass itself (netG(batch): one RANDOM timestep per image), averaged over the set."""
    netG.eval()
    torch.manual_seed(seed)
    tot, cnt = 0.0, 0
    for s in range(0, inp.shape[0], bs):
        data = dict(input=inp[s:s + bs].to(dev), target=tgt[s:s + bs].to(dev))
        b, c, h, w = data["target"].shape
        tot += (netG(data).sum() / int(b * c * h * w)).item() * b      # same normalisation as in finetune.py
        cnt += b
    return tot / cnt


def main():
    args = parse()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.finetune_py)))
    import finetune as F                                              # unchanged, frozen file
    F.LOG_PATH = os.path.join(out, "val_run%s.log" % args.run)

    sys.path.insert(0, args.repo_dir)
    os.chdir(args.repo_dir)
    import core.logger as Logger
    import core.metrics as Metrics
    import data as Data
    import model as Model
    from model.ddpm_modules import diffusion as D
    F.R.Logger, F.R.Metrics, F.R.Data, F.R.Model, F.R.D = Logger, Metrics, Data, Model, D
    D.GaussianDiffusion.set_new_noise_schedule = F.set_new_noise_schedule_mod     # the same monkeypatch as in finetune.py
    F.EXPONENT = F.X if args.run == "B" else 1.0
    F.log("post-hoc validation, run=%s exponent=%s args=%s" % (args.run, F.EXPONENT, vars(args)))

    val_set = Data.create_dataset(dict(dataroot=args.val_root, resolution=256, data_len=-1, name="EUVP-val"), "val")
    names = sorted(os.listdir(os.path.join(args.val_root, "input_256")))
    assert len(names) == len(val_set)
    items = [val_set[i] for i in range(len(val_set))]
    val_fp = hashlib.md5("".join(names).encode()).hexdigest()[:12]            # of the full folder, even in a dry run
    if args.limit > 0:
        items, names = items[:args.limit], names[:args.limit]
    inp = torch.stack([it["input"] for it in items])
    tgt = torch.stack([it["target"] for it in items])
    F.log("validation pairs: %d   fingerprint %s" % (len(names), val_fp))

    diffusion, cfg = F.build_model(args)                              # loads the PRETRAINED weights
    netG = diffusion.netG
    T = int(netG.num_timesteps)
    ts = [int((k + 0.5) * T / args.n_t) for k in range(args.n_t)]
    F.log("timesteps scored: %s" % ts)
    seed0 = args.seed * 1000 + 500000

    # ---------------- pretrained weights under this run's schedule ----------------
    info_pre = F.apply_and_check_schedule(diffusion, cfg)
    F.log("pretrained weights, schedule OK: %s" % info_pre)
    pre_params = {k: v.detach().cpu().clone() for k, v in netG.named_parameters()}
    per_pre = evaluate(netG, inp, tgt, ts, args.batch_size, seed0, "pretrained", F)
    tp_pre = [train_path_estimate(netG, inp, tgt, args.batch_size, args.seed + r) for r in range(args.n_train_path)]
    F.log("pretrained: mean loss %.5f   (training-path estimate, random t: %s)" % (per_pre.mean(), ", ".join("%.5f" % v for v in tp_pre)))

    # ---------------- fine-tuned weights ----------------
    st = torch.load(args.final_ckpt, map_location="cpu")
    netG.load_state_dict(st)                                          # strict
    del st
    # the schedule stored INSIDE final_gen.pth must be this run's schedule (catches an A file given to B or vice versa);
    # only after this check is the schedule re-applied, which would otherwise hide a mix-up
    sched = cfg["model"]["beta_schedule"]["train"]
    betas = np.asarray(F.R.D.make_beta_schedule(schedule=sched["schedule"], n_timestep=sched["n_timestep"],
                                                linear_start=sched["linear_start"], linear_end=sched["linear_end"]), dtype=np.float64)
    want_last = float(np.cumprod(1.0 - betas)[-1] ** F.EXPONENT)
    got_last = float(netG.alphas_cumprod[-1])
    F.log("schedule stored in final_gen.pth: abar_last %.4e   expected for run %s: %.4e" % (got_last, args.run, want_last))
    assert abs(got_last - want_last) < 1e-3 * want_last, "final_gen.pth was trained with a different schedule than run %s" % args.run
    info_fin = F.apply_and_check_schedule(diffusion, cfg)             # re-apply after every load, as in finetune.py
    F.log("fine-tuned weights, schedule OK: %s" % info_fin)
    num = sum(float(((v.detach().cpu() - pre_params[k]) ** 2).sum()) for k, v in netG.named_parameters())
    den = sum(float((pre_params[k] ** 2).sum()) for k in pre_params)
    rel_change = (num / den) ** 0.5
    F.log("relative weight change fine-tuned vs pretrained: %.4f%%  (must be > 0, else final_gen.pth is the pretrained file)" % (100 * rel_change))
    assert rel_change > 0, "final_gen.pth has the same weights as the pretrained checkpoint"
    per_fin = evaluate(netG, inp, tgt, ts, args.batch_size, seed0, "fine-tuned", F)
    tp_fin = [train_path_estimate(netG, inp, tgt, args.batch_size, args.seed + r) for r in range(args.n_train_path)]
    F.log("fine-tuned: mean loss %.5f   (training-path estimate, random t: %s)" % (per_fin.mean(), ", ".join("%.5f" % v for v in tp_fin)))

    # ---------------- results ----------------
    m_pre, m_fin = per_pre.mean(axis=1), per_fin.mean(axis=1)          # per-image mean over timesteps
    diff = m_fin - m_pre
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    rows = []
    for i in range(n):
        r = dict(idx=i + 1, name=names[i], pre_mean=m_pre[i], fin_mean=m_fin[i], fin_minus_pre=diff[i])
        for k, t in enumerate(ts):
            r["pre_t%d" % t] = per_pre[i, k]
        for k, t in enumerate(ts):
            r["fin_t%d" % t] = per_fin[i, k]
        rows.append(r)
    csv_path = os.path.join(out, "val_perimage_Run%s.csv" % args.run)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = dict(
        run=args.run, exponent=F.EXPONENT, n_val=n, val_fingerprint=val_fp, timesteps=ts,
        schedule_pretrained=info_pre, schedule_finetuned=info_fin,
        val_loss_pretrained_mean=float(m_pre.mean()), val_loss_pretrained_sd=float(m_pre.std(ddof=1)),
        val_loss_finetuned_mean=float(m_fin.mean()), val_loss_finetuned_sd=float(m_fin.std(ddof=1)),
        paired_diff_finetuned_minus_pretrained=float(diff.mean()), paired_diff_se=float(se),
        n_images_finetuned_lower=int((diff < 0).sum()),
        per_timestep=[dict(t=t, pretrained=float(per_pre[:, k].mean()), finetuned=float(per_fin[:, k].mean()),
                           diff=float((per_fin[:, k] - per_pre[:, k]).mean()),
                           diff_se=float((per_fin[:, k] - per_pre[:, k]).std(ddof=1) / np.sqrt(n))) for k, t in enumerate(ts)],
        training_path_estimate_pretrained=tp_pre, training_path_estimate_finetuned=tp_fin,
        relative_weight_change=rel_change,
        script_md5=hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        finetune_md5=hashlib.md5(open(os.path.abspath(args.finetune_py), "rb").read()).hexdigest())
    json.dump(summary, open(os.path.join(out, "val_summary_Run%s.json" % args.run), "w"), indent=2)

    F.log("")
    F.log("RESULT run %s (exponent %s), %d validation pairs, loss = mean |noise - predicted noise|" % (args.run, F.EXPONENT, n))
    F.log("  pretrained : %.5f  (sd over images %.5f)" % (m_pre.mean(), m_pre.std(ddof=1)))
    F.log("  fine-tuned : %.5f  (sd over images %.5f)" % (m_fin.mean(), m_fin.std(ddof=1)))
    F.log("  fine-tuned minus pretrained: %+.5f  (s.e. %.5f; lower for fine-tuned on %d of %d images)" % (diff.mean(), se, (diff < 0).sum(), n))
    F.log("  per timestep  t : pretrained  fine-tuned  difference")
    for d in summary["per_timestep"]:
        F.log("  %14d : %.5f     %.5f     %+.5f" % (d["t"], d["pretrained"], d["finetuned"], d["diff"]))
    F.log("ALL DONE")


if __name__ == "__main__":
    main()
