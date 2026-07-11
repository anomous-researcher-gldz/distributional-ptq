# From Distribution to Decision: A Diagnostic for Composable PTQ Primitives

## Abstract

Post-training quantization (PTQ) methods are typically architecture-specific. Yet the distributional pathologies they target, dense cores with sparse outliers and input-conditioned activation clusters, recur across model families. We argue PTQ should be distribution-specific instead. We propose two composable primitives: Dual-Band Affine Folding (DBAF) for outliers and Prompt-Conditioned Scale Anchoring (PCSA) for clusters. A two-statistic diagnostic dispatches them from one calibration pass. A per-tensor outlier gate decides where DBAF applies; a per-prompt compactness ratio decides where PCSA applies. The diagnostic is a selection heuristic, not a calibrated predictor. Four predictive checks confirm the dispatch: a rotation cliff on a seven-host LLaMA-3-8B W4A4 matrix, cross-family transfer to Qwen-2.5-7B, per-site gating on SAM-B (+4.8 mAP) and LLaMA-3-8B KV-cache (+5pp NIAH at 8k), and a negative SwinIR-×3 prediction at training-free W4. Both primitives are training-free and calibrate in seconds.

## Repository Structure

```
distributional-ptq/
├── ahcptq/          # AHCPTQ: SAM quantization with DBAF+PCSA
├── FlatQuant/       # FlatQuant: LLM quantization with DBAF+PCSA
├── CompSRT/         # CompSRT: Image super-resolution quantization with DBAF
├── ckpt/            # Model checkpoints (SAM)
├── exp/             # Quantization configs (SAM)
├── mmdetection/     # MMDetection (SAM detector dependency)
└── projects/        # SAM project configs
```

---

## 1. AHCPTQ — Segment Anything Model (SAM)

### 1.1 Environment Setup

```bash
# Create environment
conda create -n ahcptq python=3.7 -y
conda activate ahcptq
pip install torch torchvision

# Install MMCV
pip install -U openmim
mim install "mmcv-full<2.0.0"

# Install other requirements
pip install -r requirements.txt

# Compile CUDA operators
cd projects/instance_segment_anything/ops
python setup.py build install
cd ../../..

# Install mmdet
cd mmdetection/
python3 setup.py build develop
cd ..
```

### 1.2 Prepare Dataset

Download the [COCO](https://cocodataset.org/#download) dataset (2017 train/val + annotations):

```
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

### 1.3 Download Model Weights

Save to `ckpt/`:

| Model       | Download |
|-------------|----------|
| SAM-B       | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth (official Meta) |
| SAM-L       | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth (official Meta) |
| SAM-H       | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth (official Meta) |
| Faster-RCNN | [mmdetection model zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html) (public) |
| YOLOX       | [mmdetection model zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html) (public) |
| HDETR       | [H-Deformable-DETR](https://github.com/HDETR/H-Deformable-DETR) project (public) |
| DINO        | [DINO](https://github.com/IDEA-Research/DINO) detection release (public) |

The three SAM checkpoints are the official Meta releases (`facebookresearch/segment-anything`).
The four SAM-prompting detectors are standard public checkpoints assembled by the
Instance-Segment-Anything project on which `projects/instance_segment_anything/` is
based; that directory holds the exact configs (Faster R-CNN and YOLOX are also in the
[mmdetection model zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html)).

### 1.4 Run Experiments

```bash
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/<DETECTOR>/<MODEL.py> \
  --q_config ./exp/<QCONFIG>.yaml \
  --quant-encoder
```

Example (W4A4, SAM-B, YOLO detector):

```bash
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/yolox/yolo_l-sam-vit-b.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder
```

**Note:** For HDETR/DINO, set `keep_gpu: False` in the YAML config if memory is insufficient. See the original AHCPTQ README for details.

---

## 2. FlatQuant — Large Language Models (LLM)

### 2.1 Installation

```bash
conda create -n flatquant python=3.10 -y
conda activate flatquant
pip install -r requirements.txt && pip install -e . 
pip install flash-attn --no-build-isolation
```

**Note:** - To run models like LLaMA2 and LLaMA3, please install dependencies from `requirements_llama2.txt`.

### 2.2 Data Preparation

Download datasets in `./datasets`.

**Calibration set or PPL evaluation**

| Dataset   | Local Dir                  | URL                                                                                                                        |
| --------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| WikiText2 | ./datasets/wikitext        | [https://huggingface.co/datasets/wikitext](https://huggingface.co/datasets/wikitext)                                       |
| C4        | ./datasets/allenai/c4      | [https://huggingface.co/datasets/allenai/c4](https://huggingface.co/datasets/allenai/c4)                                   |
| Pile      | ./datasets/pile-val-backup | [https://huggingface.co/datasets/mit-han-lab/pile-val-backup](https://huggingface.co/datasets/mit-han-lab/pile-val-backup) |

**Commonsense QA evaluation**

For QA evaluation, we use local config files to specify the paths to local datasets. First, copy the dataset config files under `~/anaconda3/envs/flatquant/lib/python3.10/site-packages/lm_eval/tasks` to `./datasets/lm_eval_configs/tasks`. Next, modify the config item `dataset_path` in each QA dataset's config file to the local directory listed in the following table.

| Dataset         | Local Dir                 | URL                                                                                                                    |
| --------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| ARC-E and ARC-C | ./datasets/ai2_arc        | [https://huggingface.co/datasets/allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)                     |
| HellaSwag       | ./datasets/hellaswag      | [https://huggingface.co/datasets/Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag)                     |
| LAMBADA         | ./datasets/lambada_openai | [https://huggingface.co/datasets/EleutherAI/lambada_openai](https://huggingface.co/datasets/EleutherAI/lambada_openai) |
| PIQA            | ./datasets/piqa           | [https://huggingface.co/datasets/ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa)                               |
| WinoGrande      | ./datasets/winogrande     | [https://huggingface.co/datasets/winogrande](https://huggingface.co/datasets/winogrande)                               |

### 2.3 Model Preparation

Download models in `./modelzoo`.

| Model       | Local Dir                      | URL                                                                                                      |
| ----------- | ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| LLaMA-2-7B  | ./modelzoo/meta-llama/Llama-2-7b-hf  | [https://huggingface.co/meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)             |
| LLaMA-2-13B | ./modelzoo/meta-llama/Llama-2-13b-hf | [https://huggingface.co/meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)           |
| LLaMA-2-70B | ./modelzoo/meta-llama/Llama-2-70b-hf | [https://huggingface.co/meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)           |
| LLaMA-3-8B  | ./modelzoo/meta-llama/Meta-Llama-3-8B  | [https://huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)   |
| LLaMA-3-70B | ./modelzoo/meta-llama/Meta-Llama-3-70B | [https://huggingface.co/meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) |
| LLaMA-3-8B-Ins | ./modelzoo/meta-llama/Meta-Llama-3-8B-Instruct | [https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| LLaMA-3-70B-Ins | ./modelzoo/meta-llama/Meta-Llama-3-70B-Instruct | [https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) |
| LLaMA-3.1-8B | ./modelzoo/meta-llama/Llama-3.1-8B | [https://huggingface.co/meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| LLaMA-3.1-70B | ./modelzoo/meta-llama/Llama-3.1-70B | [https://huggingface.co/meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B) |
| LLaMA-3.1-8B-Ins | ./modelzoo/meta-llama/Llama-3.1-8B-Instruct | [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| LLaMA-3.1-70B-Ins | ./modelzoo/meta-llama/Llama-3.1-70B-Instruct | [https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) |
| LLaMA-3.3-70B-Ins | ./modelzoo/meta-llama/Llama-3.3-70B-Instruct | [https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| Qwen2.5-7B-Ins | ./modelzoo/Qwen/Qwen2.5-7B-Instruct | [https://huggingface.co/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Qwen2.5-32B-Ins | ./modelzoo/Qwen/Qwen2.5-32B-Instruct | [https://huggingface.co/Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |

### 2.4 Run Experiments

We provide full script to run FlatQuant in `./scripts/`. We use LLaMa-3-8B as an example here:

1. Weight-Activation-KV Cache Quantization

```bash
# W4A4KV4
python ./main.py \
    --model ./modelzoo/meta-llama/Meta-Llama-3-8B\
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --output_dir ./outputs --save_matrix \
    --lm_eval --lm_eval_batch_size 16
```

2. Weight-Only Quantization

```bash
# W4A16
python ./main.py \
    --model ./modelzoo/meta-llama/Meta-Llama-3-8B \
    --w_bits 4 \
    --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --output_dir ./outputs --exp_name wonly --save_matrix \
    --lm_eval --lm_eval_batch_size 16
```

3. Reproduce Evaluation Results of Our Paper
   
   1\) Download the pretrained FlatQuant parameters you want through [Model Zoo](FlatQuant/README.md#model-zoo).
   
   2\) Inference with `--reload_matrix` and `--matrix_path PATH_TO_XXX`, take LLaMa-3-8B with W4A4KV4 quantization as an example:

```bash
python ./main.py \
    --model ./modelzoo/meta-llama/Meta-Llama-3-8B \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --output_dir ./outputs --save_matrix \
    --lm_eval --lm_eval_batch_size 16 \
    --reload_matrix --matrix_path PATH_TO_XXX 
```

Use `--disable_dbaf` to run ablation experiments without DBAF.
Use `--disable_pcsa` to run ablation experiments without PCSA.

---

## 3. CompSRT — Image Super-Resolution

### 3.1 Environment Setup

```bash
cd CompSRT

# Create environment
conda create -n srtquant python=3.9 -y
conda activate srtquant

pip install six
pip install --no-cache-dir \
  torch==2.0.1+cu117 \
  torchvision==0.15.2+cu117 \
  torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
pip install -e . --no-build-isolation -v

pip install -v --no-build-isolation causal_conv1d==1.0.0
pip install -v --no-build-isolation mamba_ssm==1.0.1
```

### 3.2 Datasets

Download and place in `CompSRT/datasets/`:

* Training set (DF2K): [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) + Flickr2K (public) — the standard DF2K super-resolution training set
* Testing set: standard SR benchmarks (Set5, Set14, BSD100, Urban100, Manga109), public — e.g. via [BasicSR](https://github.com/XPixelGroup/BasicSR)
* Calibration data: regenerate from DF2K with `CompSRT/basicsr/getcalidata.py` (no separate download needed)
* Pretrained models: official [SwinIR](https://github.com/JingyunLiang/SwinIR) lightweight-SR checkpoints (e.g. `002_lightweightSR_DIV2K_*_SwinIR-S_x{2,3,4}.pth`)

### 3.3 Run Experiments

Training (with optional pruning):

```bash
# Example: 4-bit x4 SR
python basicsr/train.py -opt options/train/train_srtquant_x4.yml --pruning 0.4 \
  --force_yml bit=4 name=train_srtquant_x4_bit4
```

Testing:

```bash
python basicsr/test.py -opt options/test/test_srtquant_x2.yml --pruning 0.4 \
  --force_yml bit=4 name=test_srtquant_x2_bit4 \
  path:pretrain_network_Q=experiments/train_srtquant_x2_bit4/models/<best_model.pth>
```

See `CompSRT/README.md` for Docker/Singularity setup and statistical analysis scripts.

---

## 4. Reproducing the Paper's Results

Sections 1–3 above run the **trained/host** quantizers (AHCPTQ+DBAF+PCSA for SAM,
FlatQuant+DBAF+PCSA for LLMs, CompSRT for SR). The paper's **training-free** headline
results use dedicated drivers under `scripts/`:

| Paper result | Driver |
|---|---|
| Table 4 — LLaMA-3-8B W4A4 rotation cliff, non-rotation cells (RTN/GPTQ/AWQ/SmoothQuant ± DBAF, the 16–33 wt2 band) | `scripts/run_training_free_full_table.py` → `scripts/aggregate_host_matrix.py` (rotation cells QuaRot/FlatQuant via §2) |
| Table 19 / §4.5 — SwinIR-×3 negative prediction (−0.31 dB Set5, training-free RTN+DBAF) | `scripts/run_training_free_swinir.py` |
| Table 28 — training-free composability (RTN/GPTQ/AWQ ± DBAF / PCSA-tf × 8 models) | `scripts/run_training_free_full_table.py` |
| Training-free SAM-B + DBAF | `scripts/run_training_free_sam.py` |
| SAM-B +4.8 mAP (AHCPTQ+DBAF+PCSA) | §1 (`ahcptq/solver/test_quant.py`) |

**How to run the training-free drivers** (use the `flatquant` env from §2.1 for LLM
targets, the `ahcptq` env from §1.1 for SAM):

```bash
# Table 4 / Table 28 — one training-free cell (host × augment × model)
python scripts/run_training_free_full_table.py \
  --target llama3-8b --method rtn --augments dbaf+pcsa_tf \
  --out outputs/G8-training-free-full/llama3-8b/rtn_dbaf+pcsa_tf
# ...sweep all cells, then assemble the host matrix:
python scripts/aggregate_host_matrix.py

# §4.5 — SwinIR-×3 negative prediction (training-free RTN+DBAF)
python scripts/run_training_free_swinir.py --scale 3 \
  --pretrained CompSRT/experiments/<SwinIR-x3.pth> --dataset <Set5_HR_dir>

# Training-free SAM-B + DBAF
python scripts/run_training_free_sam.py --model-type vit_b --sam-ckpt ckpt/sam_vit_b_01ec64.pth
```

Each driver takes explicit `--` arguments (model, dataset, output paths); nothing is
hardcoded. Model/dataset locations are prepared in §1–§3.

### Author-response / reviewer-discussion evidence

The added analyses requested during review — rotation ±DBAF 2×2, CLIP/Whisper/DiT
breadth, random seeds, distribution shift, PCSA compactness site-hunt, descriptor
ablation, and the threshold/AUC robustness checks — live under
[`cross_arch_generalization/`](cross_arch_generalization/README.md). That README maps
each reviewer question to its script and committed result JSON, and gives run
instructions (a zero-GPU tier reproduces the headline robustness/AUC numbers in
seconds from the committed per-layer statistics). All result JSONs are committed under
`cross_arch_generalization/results/`.

---

## Acknowledgments

This work builds upon [AHCPTQ](https://github.com/Keio-CSG/AHCPTQ), [FlatQuant](https://github.com/ruikangliu/FlatQuant), and [CompSRT](https://github.com/anonymous-researcher-99/CompSRT).

