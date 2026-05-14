#!/usr/bin/env python3
"""
Standalone OmniQuant eval-from-checkpoint script.

Loads a saved omni_parameters.pth (calibrated LWC/LET params per layer),
re-applies quantization, and evaluates WikiText-2 (and optionally C4) PPL.

Usage:
    python scripts/run_omniquant_eval_from_ckpt.py \
        --model /data/modelzoo/meta-llama/Meta-Llama-3-8B \
        --params_path /data/outputs/.../omni_parameters.pth \
        --wbits 4 --abits 4 \
        --output /data/outputs/.../eval.json \
        [--also_eval_c4] [--let] [--group_size 128]
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Bootstrap: ensure OmniQuant and its siblings are on sys.path
# ---------------------------------------------------------------------------
OMNIQUANT_DIR = Path(__file__).resolve().parents[1] / "OmniQuant"
if str(OMNIQUANT_DIR) not in sys.path:
    sys.path.insert(0, str(OMNIQUANT_DIR))

# ---------------------------------------------------------------------------
# OmniQuant internal imports (after path setup)
# ---------------------------------------------------------------------------
from models.LMClass import LMClass
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.utils import (
    register_scales_and_zeros,
    smooth_and_quant_inplace,
    set_quant_state,
)
from datautils import get_loaders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_args_namespace(cli):
    """Construct the Namespace that LMClass / omniquant internals expect."""
    import types
    args = types.SimpleNamespace()
    args.model = cli.model
    args.batch_size = 1
    args.attn_implementation = "eager"
    args.wbits = cli.wbits
    args.abits = cli.abits
    args.group_size = cli.group_size
    args.symmetric = False
    args.disable_zero_point = False
    args.a_dynamic_method = "per_token"
    args.w_dynamic_method = "per_channel"
    args.let = cli.let
    args.lwc = True          # always true: we load bound_factor params
    args.aug_loss = False
    args.deactive_amp = False
    if (cli.wbits < 16 and cli.wbits >= 8) or (cli.abits < 16 and cli.abits >= 8):
        args.deactive_amp = True
    # net / model_family — derived the same way main.py does
    args.net = cli.model.split('/')[-1]
    args.model_family = args.net.split('-')[0]
    args.multigpu = False
    # quant param dicts
    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc": True,
        "disable_zero_point": args.disable_zero_point,
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = dict(args.act_quant_params)
    args.k_quant_params = dict(args.act_quant_params)
    args.v_quant_params = dict(args.act_quant_params)
    args.p_quant_params = {"n_bits": 16, "metric": "fix0to1"}
    return args


def load_model(cli, logger):
    """Build LMClass, move to GPU, set eval mode."""
    args = build_args_namespace(cli)
    logger.info(f"Loading model from {cli.model}")
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    for p in lm.model.parameters():
        p.requires_grad = False
    return lm, args


def apply_omniquant_with_params(lm, args, omni_parameters, logger):
    """
    Wrap each decoder layer with QuantDecoderLayer, load saved params,
    run smooth_and_quant_inplace, then register scales/zeros.
    This mirrors what omniquant() does but skips the calibration loop.
    """
    model = lm.model
    dev = lm.device

    is_llama = "llama" in args.net.lower() or "mixtral" in args.net.lower()

    if is_llama or "mixtral" in args.net.lower():
        layers = model.model.layers
        DecoderLayer = QuantLlamaDecoderLayer
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        DecoderLayer = QuantOPTDecoderLayer
    else:
        raise ValueError(f"Unsupported net: {args.net}")

    model.config.use_cache = False

    for i in tqdm(range(len(layers)), desc="Re-applying quantization"):
        layer = layers[i].to(dev)

        qlayer = DecoderLayer(model.config, layer, args)
        qlayer = qlayer.to(dev)

        # Register LET smooth params if the checkpoint contains them
        if args.let:
            # If LET was used during calibration, the params are already in
            # the state_dict; we need to register them first with dummy values
            # so load_state_dict doesn't crash on missing parameters.
            dtype = torch.float16
            qlayer.register_parameter(
                "qkt_smooth_scale",
                nn.Parameter(torch.ones(
                    layer.self_attn.q_proj.out_features, device=dev, dtype=dtype))
            )
            pairs = {"q_proj": "qkv", "o_proj": "out", "up_proj": "fc1"}
            for name, module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key, tag in pairs.items():
                        if key in name:
                            feat = module.weight.shape[0]
                            qlayer.register_parameter(
                                f"{tag}_smooth_scale",
                                nn.Parameter(torch.ones(feat, device=dev, dtype=dtype))
                            )
                            qlayer.register_parameter(
                                f"{tag}_smooth_shift",
                                nn.Parameter(torch.zeros(feat, device=dev, dtype=dtype))
                            )
                            break

        # Load the saved omni params for this layer (strict=False: ignore missing)
        if i in omni_parameters:
            missing, unexpected = qlayer.load_state_dict(omni_parameters[i], strict=False)
            if unexpected:
                logger.debug(f"Layer {i}: unexpected keys in checkpoint: {unexpected}")
        else:
            logger.warning(f"Layer {i}: no params in checkpoint, using defaults")

        qlayer.let = args.let

        # Fuse smooth transforms + quantize weights in-place
        set_quant_state(qlayer, weight_quant=False, act_quant=True)
        smooth_and_quant_inplace(qlayer, args, is_llama)

        # Register quantizer scales/zeros needed at forward time
        register_scales_and_zeros(qlayer)

        layers[i] = qlayer.to("cpu")
        torch.cuda.empty_cache()

    model.config.use_cache = True
    logger.info("Quantization re-applied from checkpoint.")


@torch.no_grad()
def eval_ppl(lm, dataset_name, cache_dir, logger):
    """Evaluate perplexity on dataset_name ('wikitext2' or 'c4')."""
    dev = lm.device
    args_net = lm.args.net
    args_model = lm.args.model
    model_family = lm.args.model_family
    seqlen = lm.seqlen

    cache_testloader = os.path.join(cache_dir, f"testloader_{model_family}_{dataset_name}_all.cache")
    if os.path.exists(cache_testloader):
        testloader = torch.load(cache_testloader)
        logger.info(f"Loaded cached testloader from {cache_testloader}")
    else:
        _, testloader = get_loaders(
            dataset_name,
            seed=2,
            model=args_model,
            seqlen=seqlen,
        )
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(testloader, cache_testloader)

    if "c4" in dataset_name:
        testenc = testloader
    else:
        testenc = testloader.input_ids

    nsamples = testenc.numel() // seqlen

    use_cache = lm.model.config.use_cache
    lm.model.config.use_cache = False
    lm.model.eval()
    lm.model = lm.model.to(dev)

    nlls = []
    loss_fct = nn.CrossEntropyLoss()

    for i in tqdm(range(nsamples), desc=f"PPL eval [{dataset_name}]"):
        batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(dev)

        if "opt" in args_net.lower():
            outputs = lm.model.model.decoder(batch)
        elif "llama" in args_net.lower() or "mixtral" in args_net.lower():
            outputs = lm.model.model(batch)
        else:
            raise ValueError(f"Unsupported net: {args_net}")

        hidden_states = outputs[0]
        logits = lm.model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :]
        shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][:, 1:].to(
            lm.model.lm_head.weight.device
        )
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    lm.model.config.use_cache = use_cache
    logger.info(f"{dataset_name} PPL: {ppl.item():.4f}")
    return ppl.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OmniQuant eval-from-checkpoint")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the HF model directory")
    parser.add_argument("--params_path", type=str, required=True,
                        help="Path to omni_parameters.pth")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--let", action="store_true",
                        help="Set if LET (learnable equivalent transformation) was used during calibration")
    parser.add_argument("--also_eval_c4", action="store_true",
                        help="Also evaluate C4 perplexity")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write JSON results")
    parser.add_argument("--cache_dir", type=str, default=str(OMNIQUANT_DIR / "cache"),
                        help="Cache dir for dataset loaders")
    cli = parser.parse_args()

    # Logger
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    t0 = time.time()

    # 1. Load model
    lm, args = load_model(cli, logger)

    # 2. Load omni_parameters.pth
    logger.info(f"Loading checkpoint from {cli.params_path}")
    omni_parameters = torch.load(cli.params_path, map_location="cpu")
    logger.info(f"Checkpoint contains params for {len(omni_parameters)} layers")

    # 3. Re-apply quantization
    apply_omniquant_with_params(lm, args, omni_parameters, logger)

    # 4. Eval WikiText-2
    wikitext2_ppl = eval_ppl(lm, "wikitext2", cli.cache_dir, logger)

    # 5. (Optional) Eval C4
    c4_ppl = None
    if cli.also_eval_c4:
        c4_ppl = eval_ppl(lm, "c4", cli.cache_dir, logger)

    wallclock = time.time() - t0

    # 6. Write JSON
    results = {
        "arm": os.path.basename(os.path.dirname(cli.params_path)),
        "model": cli.model,
        "params_path": cli.params_path,
        "wbits": cli.wbits,
        "abits": cli.abits,
        "group_size": cli.group_size,
        "let": cli.let,
        "wikitext2_ppl": wikitext2_ppl,
        "c4_ppl": c4_ppl,
        "wallclock_sec": round(wallclock, 1),
    }
    Path(cli.output).parent.mkdir(parents=True, exist_ok=True)
    with open(cli.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {cli.output}")
    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
