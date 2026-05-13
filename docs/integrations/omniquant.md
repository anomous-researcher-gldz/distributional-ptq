# OmniQuant DBAF + PCSA Integration Notes

## Identified hooks

### Weight quantization

- Function: `quantize.quantizer.UniformAffineQuantizer.forward`
- Path: `OmniQuant/quantize/quantizer.py:118`
- Signature: `def forward(self, x: torch.Tensor) -> torch.Tensor`
- What it does: Computes per-token/per-channel dynamic scale+zero-point then calls `self.fake_quant(x, self.scale, self.round_zero_point)` to produce a fake-quantized (dequantized) weight or activation tensor.
- DBAF wrap point: wrap **before** the `self.fake_quant(...)` call at line 129; fold the DBAF gate factor into `x` entering `fake_quant`, and unfold (divide) the returned `x_dequant` before returning from `forward`.

The weight quantizer is actually called in two places:
1. **Training loop** — `smooth_and_quant_temporary` (`quantize/utils.py:96`) calls `module.weight_quantizer(module.temp_weight)`, setting `module.use_temporary_parameter = True` so the quantized value is cached as `temp_weight`.
2. **Inplace finalization** — `smooth_and_quant_inplace` (`quantize/utils.py:135`) calls `module.weight_quantizer(module.weight)` and stores the result directly back to `module.weight`.

Both paths funnel through `UniformAffineQuantizer.forward`, so a single monkey-patch of that `forward` method is sufficient.

### Activation quantization

- Function: `quantize.quantizer.UniformAffineQuantizer.forward`
- Path: `OmniQuant/quantize/quantizer.py:118`
- Signature: `def forward(self, x: torch.Tensor) -> torch.Tensor`
- What it does: Same `UniformAffineQuantizer.forward` — the same class instance is used for activations; OmniQuant distinguishes weight vs. activation quantizers only by which `QuantLinear` attribute holds the instance (`weight_quantizer` vs. `act_quantizer`).
- DBAF wrap point: wrap **before** `fake_quant` call at line 129, folding the gate into `x` and unfolding after.
- PCSA-tf insertion point: **before** the `per_token_dynamic_calibration` call at line 125; supply the per-prompt learned scale `s_prompt` by premultiplying `x` with it (or by overriding `self.scale` after calibration). The natural hook is overriding `per_token_dynamic_calibration` on the `act_quantizer` instance to inject the PCSA routing scale before the min/max range is computed.

The activation quantizer is invoked inside `QuantLinear.forward` (`quantize/int_linear.py:60`):
```
if self.use_act_quant and not self.disable_input_quant:
    input = self.act_quantizer(input)
```
A second set of activation quantizers lives in `QuantMatMul` (`quantize/int_matmul.py`), used for QKT and PV products; those are accessed via `self.qkt_matmul.x1_quantizer` etc. and go through the same `UniformAffineQuantizer.forward`.

### Calibration loop entry

- File: `OmniQuant/quantize/omniquant.py:42`
- Function: `omniquant(lm, args, dataloader, act_scales, act_shifts, logger=None)`
- Where we'd install patches and fit PCSA-tf: At the **top of the per-layer loop** (`omniquant.py:192`), immediately after `qlayer = DecoderLayer(lm.model.config, layer, args)` is constructed (line 203) and before `set_quant_state` is called (line 208). At this point every `QuantLinear` and `QuantMatMul` submodule has been instantiated and their `act_quantizer` / `weight_quantizer` attributes are accessible. We can:
  1. Replace `qlayer`'s `act_quantizer.forward` methods (or subclass `UniformAffineQuantizer`) with DBAF-wrapped versions.
  2. Register PCSA-tf `nn.Parameter` objects on each `act_quantizer` and expose them via `lwc_parameters` / `get_omni_parameters` so OmniQuant's existing AdamW optimizer loop trains them automatically.
  3. Alternatively, run a separate PCSA-tf fit pass over the same `nsamples` calibration inputs **after** the OmniQuant epoch loop completes (after line 277).

### Per-prompt descriptor extraction

- During calibration, OmniQuant feeds `args.nsamples` (default 128) WikiText-2 segments of length 2048 through the model one by one. The inputs to every layer are pre-captured into the `inps` tensor (shape `[nsamples, seqlen, hidden_size]`) at lines 109–139 of `omniquant.py`.
- Hidden-state hook point for descriptor extraction: `QuantLlamaDecoderLayer.forward` at `models/int_llama_layer.py:237` — the `hidden_states` argument entering `forward` is the post-residual, pre-layernorm activation for that layer (the "attention entry" representation). This is already stored sample-by-sample in `quant_inps[j]` / `fp_inps[j]` during the per-layer calibration loop.
- Each sample produces one descriptor by mean-pooling: mean-pool `hidden_states` over the sequence dimension (`dim=1`) to obtain a `[hidden_size]` descriptor for sample `j`. This can be extracted inside the calibration loop at `omniquant.py:285` (the `quant_inps` update loop) by hooking into `qlayer.forward` or by reading `quant_inps[j]` directly before and after the forward pass.

## Recommended monkey-patch strategy

The cleanest approach is to subclass `UniformAffineQuantizer` into `DBAFUniformAffineQuantizer` that adds learnable `gate` and `scale_route` parameters (for DBAF and PCSA-tf respectively), override `forward` to fold/unfold the gate around `fake_quant` and premultiply the input with the PCSA route scale before `per_token_dynamic_calibration`, then at the top of the per-layer loop in `omniquant.py` walk `qlayer.named_modules()` and replace every `QuantLinear.act_quantizer` and `QuantLinear.weight_quantizer` instance with the new subclass using `setattr`. Weight quantizer and activation quantizer replacements can be distinguished by attribute name, so separate DBAF gate parameterizations can be applied. No changes to OmniQuant source files are needed.

## Smoke-test plan

Run OmniQuant with `--epochs 1 --nsamples 4 --wbits 4 --abits 4 --let --lwc` on a single Llama-2-7b layer by patching the `layers` slice to `layers[:1]` (or by setting `len(layers)` to 1 via a monkey-patch before the loop), confirming that the DBAF-wrapped `forward` executes without NaN and that the PCSA-tf `scale_route` parameter appears in `get_omni_parameters` output and receives a non-zero gradient after the first optimizer step. A zero-ablation (gate=1, route=1 initialization) should reproduce the baseline OmniQuant loss curve for layer 0 within floating-point noise.
