# G7: 2DQuant SwinIR Calibration Driver — Status

## Actual 2DQuant CLI invocation pattern

```bash
# From 2DQuant/scripts/2DQuant_test.sh
python basicsr/test.py -opt options/test/test_2DQuant_x<SCALE>.yml \
    --force_yml bit=4 name=<run_name> \
                path:pretrain_network_Q=<ckpt_path> \
                pathFP:pretrain_network_FP=<ckpt_path>
```

Key notes:
- `--force_yml` accepts `key=value` or `nested:key=value` (colon-separated nesting)
- `basicsr/test.py` resolves to `2DQuant/basicsr/test.py` (run from `2DQuant/` cwd)
- Results written to `2DQuant/results/<run_name>/` (controlled by `opt['name']`)
- Metrics are logged to `results/<run_name>/test_<name>_<timestamp>.log` as plaintext
- No native `--output_root` flag; driver script harvests PSNR/SSIM from log and emits eval.json

### ARM wiring

Arms B/C/D use `python -u -c "... runpy.run_path('basicsr/test.py', ...)"` so that
`twodquant_dbaf_pcsa_patch` is imported (and monkey-patches installed) before
`basicsr.archs.quant_arch` classes are used. The patch file lives at
`2DQuant/twodquant_dbaf_pcsa_patch.py` (already in place).

Arms C/D use synthetic pilot descriptors (`embed_dim=60`, N=128) for the PCSA-tf fit.
Real descriptors via `conv_first` hook is a follow-up integration task.

---

## YAML configs: existing vs. needed

| Config file | Status | Notes |
|---|---|---|
| `options/test/test_2DQuant_x2.yml` | **EXISTS** | bit=4 default; cali_data path hardcoded |
| `options/test/test_2DQuant_x3.yml` | **EXISTS** | bit=4 default |
| `options/test/test_2DQuant_x4.yml` | **EXISTS** | bit=4 default |

No new YAML files are needed. The driver overrides `bit=4`, `name`, and both checkpoint
paths via `--force_yml`. W4A4 is the default in all three files.

### Fields controlled by --force_yml in driver

| Field | Override |
|---|---|
| `bit` | `4` (W4A4) |
| `name` | `2DQuant_x{S}_w4a4_arm{A}` |
| `path:pretrain_network_Q` | `/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{S}.pth` |
| `pathFP:pretrain_network_FP` | same ckpt |

---

## Calibration data

2DQuant requires pre-generated per-scale `.pth` calibration batches at:
```
2DQuant/keydata/cali_data_x{2,3,4}.pth
```

Generated via:
```bash
python basicsr/train_ptq_getcalidata.py -opt options/train/train_getcalidata_x{S}.yml
```

These scripts pull **32 LR/HR patch pairs** from the DF2K training set
(`datasets/DF2K/HR` + `datasets/DF2K/LR_bicubic/X{S}`). The driver
auto-generates them if missing and `SKIP_CALI_DATA_GEN != 1`.

**Blocker**: DF2K is not present at `2DQuant/datasets/DF2K/` (not downloaded).
Options when GPU is free:
1. Download DF2K (~7 GB) and run `train_ptq_getcalidata.py` once per scale.
2. OR generate synthetic cali-data from existing Set5 HR images using a short
   custom script (32 random 128×128 crops from Set5 HR, downsampled by bicubic).
   Set5 HR is available at `/home/ubuntu/unifying-ptq/data/sr_testsets/Set5_HR`.
   The cali-data format is `{'lq': Tensor[32,3,64,64], 'gt': Tensor[32,3,H,W]}`.

---

## Benchmark datasets

The YAMLs reference Set5/Set14/B100/Urban100/Manga109 under `datasets/benchmark/`.
These need to be symlinked or copied from `/home/ubuntu/unifying-ptq/data/sr_testsets/`:
```bash
mkdir -p 2DQuant/datasets/benchmark
ln -s /home/ubuntu/unifying-ptq/data/sr_testsets/Set5_HR 2DQuant/datasets/benchmark/Set5/HR
# ... (need matching LR subdirs too)
```

---

## Checkpoint paths

SwinIR-S checkpoints (available locally):
```
/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth
/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth
/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth
```

The driver injects these via `--force_yml path:pretrain_network_Q=...` and
`pathFP:pretrain_network_FP=...`.

---

## Estimated time per cell

From 2DQuant paper / README: no explicit claim found. Based on comparable SR
quantization pipelines (BRECQ, QDROP on SwinIR-S):
- Calibration forward pass: ~5–10 s (single batch of 32 patches)
- Inference on 5 benchmarks (Set5+Set14+B100+Urban100+Manga109 ≈ 310 images at ×2):
  ~2–5 min per scale on an A100/3090

**Estimate: ~5–10 min per cell, ~60–120 min total for 12 cells.**
(TBD until first smoke run.)

---

## Blockers before running

1. **Calibration data** (`keydata/cali_data_x{2,3,4}.pth`) — not present.
   Need DF2K or synthetic alternative (see above).
2. **Benchmark LR directories** — Set5/Set14/B100/Urban100/Manga109 LR bicubic
   subdirs must exist under `2DQuant/datasets/benchmark/`. HR directories can
   be symlinked from `/home/ubuntu/unifying-ptq/data/sr_testsets/`, but LR
   bicubic images may need separate download.
3. **Local GPU is currently busy** (OmniQuant Arm A). Run after OmniQuant finishes.
4. **PCSA-tf arms (C/D)** use synthetic pilot descriptors. Real conv_first hook
   integration is a follow-up; pilot establishes the eval harness.

---

## Driver script

`/home/ubuntu/unifying-ptq/scripts/run_2dquant_swinir.sh`

Syntax verified: `bash -n` passes cleanly.
