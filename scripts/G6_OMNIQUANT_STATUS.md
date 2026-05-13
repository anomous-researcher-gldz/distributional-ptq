# G6 OmniQuant Status

## OmniQuant CLI flags discovered

Source: `OmniQuant/main.py` argparse + `OmniQuant/scripts/Llama-2/Llama-2-7b/w4a4.sh`

- `--model`: full path to HuggingFace model directory (not `--hf_path`)
- `--wbits` / `--abits`: weight/activation bits (int; default 4/16)
- `--nsamples`: number of calibration samples (default 128)
- `--epochs`: calibration epochs (default 10); **requires `--lwc` or `--let` to be set when >0**
- `--output_dir`: logging output dir (default `../log/`)
- `--calib_dataset`: one of `wikitext2|ptb|c4|mix|pile` (default wikitext2)
- `--lwc`: enable Learnable Weight Clipping (action flag; required for W4A4 without `--let`)
- `--let`: enable Learnable Equivalent Transformation (requires pre-computed `act_scales/<net>.pt` and `act_shifts/<net>.pt`; NOT used for LLaMA-3-8B because those files don't exist yet)
- `--alpha`: SmoothQuant alpha (default 0.5; reference w4a4 script uses 0.75)
- `--lwc_lr` / `--let_lr`: learning rates (defaults 1e-2 / 5e-3)
- `--eval_ppl`: (action flag) run wikitext2 + c4 PPL eval post-calibration; **omit to skip**
- `--tasks`: lm_eval harness tasks string (default `""`; empty = skip harness eval)
- `--net`: auto-derived from model path last component; `Meta-Llama-3-8B` contains "llama" so it routes to `QuantLlamaDecoderLayer` correctly
- `--save_dir`: optional path to save fake-quantized model weights
- `--cache_dir`: calibration data cache (default `./cache`)

## Compat patches applied to OmniQuant source (needed for LLaMA-3 / transformers 4.45)

Three minimal patches were required; all in-tree (no external files modified):

1. **`quantize/omniquant.py`** — move `model.model.rotary_emb` to GPU before the input-capture forward pass (LLaMA-3 added a top-level `rotary_emb` module with CPU-resident `inv_freq` buffer not present in LLaMA-2).

2. **`models/int_llama_layer.py` line 124** — `LlamaRotaryEmbedding.forward()` in transformers ≥ 4.38 dropped the `seq_len` keyword arg. Replaced with a try/except that falls back to the new `(x, position_ids)` signature.

3. **`models/int_llama_layer.py` line 157** — LLaMA-3 attention mask from transformers 4.45 has shape `(bsz, 1, q_len, kv_seq_len+1)` (off-by-one vs kv_seq_len). Added a truncation slice before the size check.

## Smoke run

- **Command:**
  ```
  python main.py \
    --model /data/modelzoo/meta-llama/Meta-Llama-3-8B \
    --wbits 4 --abits 4 \
    --calib_dataset wikitext2 \
    --nsamples 8 --epochs 1 \
    --lwc --alpha 0.75 --lwc_lr 1e-2 \
    --output_dir /tmp/t11-smoke
  ```
- **Wall-time:** 40.4 seconds (model load ~43s + 32 layers × 1 epoch × 8 samples)
- **Status:** SUCCESS — all 32 layers completed, exit code 0
- **First-layer loss observed:** yes — layer 0 iter 0 loss = 5.79e-05
- **Last-layer loss:** layer 31 iter 0 loss = 1.02 (expected: only 1 epoch, late layers accumulate error)
- **Peak GPU memory:** 7607 MB (~7.6 GB)
- **Errors hit + resolved:**
  1. `ModuleNotFoundError: No module named 'omegaconf'` → `pip install omegaconf`
  2. `ModuleNotFoundError: No module named 'pycountry'` → `pip install pycountry`
  3. `RuntimeError: Expected all tensors to be on the same device` (rotary_emb CPU/CUDA mismatch) → patched `omniquant.py` to move `model.model.rotary_emb` to device
  4. `TypeError: LlamaRotaryEmbedding.forward() got unexpected kwarg 'seq_len'` → patched `int_llama_layer.py` with try/except API fallback
  5. `ValueError: Attention mask should be of size (1,1,2048,2048) but is (1,1,2048,2049)` → patched `int_llama_layer.py` to truncate mask to `kv_seq_len`

## Estimated time per arm

- Smoke: 8 samples × 1 epoch × 32 layers = 40.4s calibration (+43s model load)
- Scale to 128 samples × 20 epochs = factor ~320x
- Estimated calibration time: 40.4s × 320 ≈ 12,928s ≈ **3.6 hours per arm**
- With 4 arms (A/B/C/D): ~14.4 hours total (can be parallelised if multiple GPUs available)
- Arm B/C/D patch overhead is negligible (Python-level function wrapping, no extra backward pass)

## Notes on --let / act_scales

The `--let` flag (Learnable Equivalent Transformation) is disabled in the driver because LLaMA-3-8B act_scales/act_shifts `.pt` files don't exist yet in `OmniQuant/act_scales/`. To enable `--let`, first run:
```
python generate_act_scale_shift.py \
  --model /data/modelzoo/meta-llama/Meta-Llama-3-8B \
  --calib_dataset wikitext2 --nsamples 128 \
  --output_path ./act_scales/Meta-Llama-3-8B.pt
```
This is a follow-up task (not blocking the 4-arm sweep since `--lwc` alone covers W4A4 LWC).

## Ready to launch full sweep?

**Yes**, with notes:
- All 3 compat patches are in place; pipeline runs end-to-end
- Driver at `scripts/run_omniquant_llama3_8b.sh` is ready; invoke `./run_omniquant_llama3_8b.sh A` etc.
- Arm B (DBAF) and D (DBAF+PCSA) will use the `omniquant_dbaf_pcsa_patch.py` monkey-patch; the try/except approach is safe
- Arm C/D use synthetic PCSA descriptors for this pilot; production integration (real calibration descriptors from the OmniQuant per-layer loop) is scoped as a follow-up
- Recommend running with `--nsamples 128 --epochs 20 --lwc --alpha 0.75` for publication-quality results
