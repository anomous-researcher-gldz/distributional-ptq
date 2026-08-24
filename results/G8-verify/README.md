# G8-verify: corroboration of the training-free SwinIR x2 cells

Run 2026-08-24 to close a provenance gap on the vision rows of the
training-free composability table (`tab:trainfree-vision` in the EMNLP
camera-ready; formerly the vision columns of `tab:trainfree`).

**These runs corroborate the published values. They are not their source** —
the original runs were never archived, and `PROVENANCE.md` lists neither table
among the JSON-backed ones.

## Result

| arm | published | here | delta |
|---|---|---|---|
| RTN (vanilla) — control | 32.61 | 32.60489 | -0.005 |
| + DBAF — control | 32.76 | 32.76274 | +0.003 |
| + PCSA-tf | 31.92 | 31.93538 | +0.015 |
| + DBAF + PCSA-tf | 32.11 | 32.0678 (mean of 4) | -0.042 |

Set5 PSNR, W4 weight-only, RTN weight host. The two controls are the ones with
existing sources (`PAPER_RESULTS.md`), and they reproduce to within 0.005 dB —
that is what makes the other two interpretable.

RTN+PCSA-tf lands at 31.94, not the 32.605 that a copy of the no-PCSA run would
give, so the published 31.92 is a genuine PCSA measurement. Its near-collision
with `results/F2-swinir-dual-gate/x2_A_w4a4_nodbaf_Set5.json` (31.9233, RTN
W4A4 no DBAF) is the coincidence the table caption predicts: PCSA-tf's damage
here is dominated by the INT4 activation quantization it introduces, so it
lands where plain W4A4 lands.

## The anchor fit is not seeded

`fit_pcsa_tf` clusters descriptors with `_kmeans`, which initialises centroids
via `torch.randperm(N)[:k]` (`FlatQuant/flatquant/baselines/pcsa_tf.py:22`);
nothing in the driver path calls `manual_seed`. So PCSA rows carry a
run-to-run component. Four refits of `+DBAF+PCSA-tf`
(`rtn_dbaf+pcsa_tf/eval.json` plus `rtn_dbaf+pcsa_tf_repeats/refit{1,2,3}.json`):

    32.05311  32.05915  32.09471  32.06408
    mean 32.0678   s.d. 0.0185   range 0.0416

The published 32.11 sits at the upper edge of that band (+0.015 above the
highest of four). Consistent with the spread; not adjudicable at n=4. The
+0.19 dB DBAF recovery claim is unaffected — measured +0.132 dB, an order of
magnitude above the spread.

## Reproducing

    python scripts/run_training_free_full_table.py \
      --target swinir-x2 --method rtn --augments {alone,dbaf,pcsa_tf,dbaf+pcsa_tf} \
      --out results/G8-verify/swinir-x2/rtn_<augments>/eval.json --force

Two traps:

1. **PCSA calibration fails silently.** `_collect_swinir_pcsa_state` globs
   `{dir}/*.png` with no `HR/` fallback, unlike `evaluate()`. Given a directory
   whose images sit in an `HR/` subdir it finds none, fits nothing, returns an
   identity state (`scales=ones`), and reports a result bit-identical to the
   no-PCSA arm — without erroring. Confirm the line
   `[driver] SwinIR PCSA-tf fitted: K=8` appears; if it does not, the run is void.
2. **`ckpt/` and `data/` are not in this repo.** The driver expects
   `ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth` and
   `data/sr_testsets/{Set5_HR,Set5_LR_x2,Urban100_HR,Urban100_LR_x2}`, with the
   HR directories holding PNGs flat (see trap 1). The vendored
   `CompSRT/basicsr/` additionally lacks `data/`, `archs/quip_hadamard.py` and
   `archs/edsr_arch.py`, and `models/__init__.py` eagerly imports a MambaIRv2
   model requiring `mamba_ssm`; an import-only stub suffices.
