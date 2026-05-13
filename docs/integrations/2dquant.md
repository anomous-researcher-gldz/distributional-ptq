# 2DQuant DBAF + PCSA-tf Integration Notes

## Identified hooks

### Weight quantization
- Class/function: `basicsr.archs.quant_arch.FakeQuantizerWeight`
- File: `2DQuant/basicsr/archs/quant_arch.py:166`
- Signature: `def forward(self, x: torch.Tensor) -> torch.Tensor`
- What it does: On first call (calibration pass) runs DOBI to find optimal `[lower_bound, upper_bound]`; on subsequent calls fake-quantizes `x` using the learned asymmetric affine mapping `s = (ub - lb) / (2^n - 1)`, returning `s * round((clip(x, lb, ub) - lb) / s) + lb`.
- DBAF wrap point: After the `if not self.calibrated` branch (i.e., once `self.calibrated = True`), wrap the core quantization expression with DBAF gate + fold/unfold around the `self.clip` / `self.round` block. In practice, patch `FakeQuantizerWeight.forward` so that when `self.calibrated` is True and the outlier gate fires, `x` is first folded with `fold_outliers`, passed through the original arithmetic, then `unfold_outliers` is applied to the result before returning.

### Activation quantization
- Class/function: `basicsr.archs.quant_arch.FakeQuantizerAct`
- File: `2DQuant/basicsr/archs/quant_arch.py:219`
- Signature: `def forward(self, x: torch.Tensor) -> torch.Tensor`
- What it does: Same DOBI-calibrated affine fake-quantizer as `FakeQuantizerWeight` but also supports `dynamic` (per-sample min/max) and `running_stat` (EMA moving average) modes for activation range estimation.
- DBAF wrap point: Identical to weight quantizer — patch `FakeQuantizerAct.forward` to fold/unfold outliers around the affine arithmetic after calibration completes.
- PCSA-tf insertion point: Inside `FakeQuantizerAct.forward`, before computing `s = (ub - lb) / (2^n - 1)`, pre-clamp `x` to the anchor-specific range `[-anchor_scale, anchor_scale]` when `_CURRENT_DESCRIPTOR` is set. This tightens the effective dynamic range to the PCSA-fitted per-cluster bound without touching `self.lower_bound` / `self.upper_bound`.

The `QuantLinear.forward` (line 466) explicitly calls `self.act_quantizer(x)` then `self.weight_quantizer(self.weight)`, so all quantization passes through these two classes. `QuantConv2d.forward` (line 584) has the same pattern. Standalone `FakeQuantizerAct` instances are also created for attention Q/K/V/attention-map quantization (`mymodule_q`, `mymodule_k`, `mymodule_v`, `mymodule_a`) inside `TDQuantModel.build_quantized_network` (2dquant_model.py:198–206) and stored in `self.quant_act`.

### Calibration loop entry
- File: `2DQuant/basicsr/models/2dquant_model.py:94`
- Function/class: `TDQuantModel.calibration`
- Where we install patches and fit PCSA-tf: The calibration runs as a single forward pass of `self.net_Q` on the pre-saved calibration batch `self.cali_data['lq']` (line 96). All `FakeQuantizerBase` submodules that have `calibrated=False` run DOBI on the first call and set `calibrated=True` before returning. The correct installation point is immediately before line 97 (`_ = self.net_Q(lq)`): call `install_dbaf_patches()` to wrap `FakeQuantizerWeight.forward` and `FakeQuantizerAct.forward`, then run the single calibration forward pass so DOBI sets all bounds, then collect descriptors and activations from a second forward pass, call `fit_pcsa_tf_on_calib_data(descs, acts, K)`, and finally call `install_pcsa_tf()`. Alternatively, monkey-patch `TDQuantModel.calibration` directly to inject this sequence.

### Per-prompt descriptor (per-image for SR)
- LR image is fed to `net_Q`; descriptor = mean-pooled feature after `SwinIR.conv_first` (line 1146 of `swinir_arch.py`). `conv_first` is a `nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)` that maps the raw LR image to the embedding space. The output shape is `[B, embed_dim, H, W]`; global-average-pooling over `(H, W)` yields a `[B, embed_dim]` descriptor per image.
- Hook point: register a `forward_hook` on `net_Q.conv_first` (the bare `SwinIR` module), capturing the output tensor and mean-pooling it with `output.mean(dim=[2, 3])` to produce the per-image descriptor.

## Recommended monkey-patch strategy

The cleanest approach mirrors OmniQuant: patch `FakeQuantizerWeight.forward` and `FakeQuantizerAct.forward` at the module-class level (not per-instance) using Python method replacement, wrapping the calibrated-path arithmetic with DBAF gate + fold/unfold; for PCSA-tf, add a second wrapper on `FakeQuantizerAct.forward` that pre-clamps `x` to the per-cluster anchor range before the original quantization arithmetic, routing via `route_pcsa_tf(_CURRENT_DESCRIPTOR, _PCSA_STATE)`. Unlike OmniQuant, 2DQuant uses **separate** classes for weights (`FakeQuantizerWeight`) and activations (`FakeQuantizerAct`), so PCSA-tf routing can be applied exclusively to `FakeQuantizerAct` with no need to discriminate by `sym` flag. Both patches are safe to apply before the DOBI calibration pass because the `if not self.calibrated` early-return path is left untouched.

## Smoke-test plan

After patching, run a single calibration forward pass on a batch of 8 random `[1, 3, 64, 64]` LR images (no real checkpoint needed — disable `self.calibrated` checks by not loading a pretrained model), confirm that `FakeQuantizerWeight.forward` and `FakeQuantizerAct.forward` are called without NaN, and that `fit_pcsa_tf_on_calib_data` populates `_PCSA_STATE` with `K` anchors. A zero-ablation (gate always False, anchor scale = inf) must reproduce the original DOBI fake-quantized output within floating-point noise.
