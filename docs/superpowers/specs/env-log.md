2026-05-12 torchao installed: torchao==0.9.0
2026-05-12 ahcptq-old env built: torch 1.13.1+cu117, mmcv-full 1.7.0, mmdet 2.26.0 vendored; CUDA ops compiled with TORCH_CUDA_ARCH_LIST=8.0 + _check_cuda_version monkey-patch (system CUDA 12.0 vs torch cu117 mismatch); sam2 1.0 installed --no-deps from source; numpy<2
2026-05-13 remote-gpu provisioned: verb-workspace (A100-SXM4-80GB 80GB), miniconda 26.3.2 + unifyptq (torch 2.6.0+cu124, torchao 0.9.0) + ahcptq-old (torch 1.13.1+cu117, mmcv 1.7.0, mmdet 2.26.0, MSDA_OK, AHCPTQ_OK)
2026-05-13 data downloads complete: COCO train2017 (118k imgs, 19GB), SR testsets (Set5/Set14/BSD100/Urban100 HR+LR_x2/x3/x4 from eugenesiow HF, 470MB), DIV2K train/valid HR+LR_bicubic_X2/X3/X4 (5.4GB), WikiText-2 HF cached
