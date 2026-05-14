# CompSRT: Quantization and Pruning for Image Super-Resolution Transformers

## Abstract

Model compression has become important for image super resolution, but the gap to full precision remains large and a deeper understanding of compression theory is needed. Prior LLM-quantization work uses Hadamard transformations to reduce outliers but explains them informally (incoherence, central limit theorem, kurtosis), without quantitative tests on which property of the post-Hadamard distribution drives the gain. We provide the first paired statistical decomposition of this question on both weights and activations of SwinIR-light. Under a Sylvester+padding implementation, the post-Hadamard distribution shows significantly reduced range, increased mass within $[-\varepsilon,\varepsilon]$, and improved normality; under an alternative QuIP#-Paley implementation that avoids zero-padding for the dominant tensor dimensions, range reduction and normality replicate while the in-band-mass effect is partially construction-dependent — so range reduction is the kernel-independent driver. Based on these findings, we introduce CompSRT, which combines Hadamard-based quantization with a scalar decomposition that adds two trainable parameters per layer to decouple the gradient pathways for the quantization scale and zero offset. Our quantization statistically significantly surpasses SOTA with gains as large as 1.53 dB and visibly improves visual quality at all bitwidths. At 3–4 bits we prune 40% of weights at ~0.4–0.5 dB cost relative to our unpruned method, achieving 6.67–15% fewer bits per parameter; on Set5 ×4 our 3-bit pruned model still *outperforms* CondiQuant (31.74 vs. 31.62 dB) and our 4-bit pruned model matches CondiQuant within 0.07 dB.

---


## Setup 

> With Conda 

```bash
# clone
git clone https://github.com/anonymous-researcher-99/CompSRT.git
cd CompSRT

# conda env
conda create -n srtquant python=3.9 -y
conda activate srtquant
#optional: set paths for cuda
#export CUDA_HOME=/usr/local/cuda-11.7
#export PATH=$CUDA_HOME/bin:$PATH
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
pip install six
pip install --no-cache-dir \
  torch==2.0.1+cu117 \
  torchvision==0.15.2+cu117 \
  torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt

pip install -e . --no-build-isolation -v

#pip install wheel
#version mismatch fix
#pip uninstall -y numpy opencv-python opencv-python-headless transformers tokenizers huggingface-hub causal-conv1d mamba-ssm 

#pip install opencv-python==4.9.0.80
#pip install numpy==1.24.3
#pip install transformers==4.37.1 tokenizers==0.15.1 huggingface-hub==0.20.3
#pip install causal_conv1d==1.0.0 --no-build-isolation
#pip install mamba_ssm==1.0.1 --no-build-isolation

pip install -v --no-build-isolation causal_conv1d==1.0.0

pip install -v --no-build-isolation mamba_ssm==1.0.1
```
> With Docker & Singularity 

```bash
# clone
git clone https://github.com/anonymous-researcher-99/CompSRT.git
cd CompSRT

# create docker environment
docker buildx build --no-cache --memory=48g --platform linux/amd64 -t compsrt:image --output=type=docker,dest=compsrt_image.tar .

# create singularity environment 
singularity build compsrt_image.sif docker-archive:path/to/compsrt_image.tar
```

## Datasets
Download:

   * [Training set (DF2K)](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link) and place them in `datasets/`
   * [Testing set](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) and place them in `datasets/`
   * [Calibration data](https://drive.google.com/file/d/1UxgyQWrToZHxsMrPursuMBtyCcNjFwUA/view?usp=drive_link)  
   * [Pretrained models](https://drive.google.com/file/d/12g_64n-hhJJbvd6cpU7VakxruGRpzhP-/view?usp=drive_link) 
   * [weights_and_activations](https://drive.google.com/file/d/1S9Vi8IyjmCY3ymmanyEDSDVY7MHAHRm5/view?usp=share_link) 

Weights and activations data is for running the statistical analysis for pre/post Hadamard transformation for the x2 model. 


## Training (w/ optional pruning)
> Requires 48GB of memory.
> To not prune, set --pruning to 0.0  

> Without slurm 
```bash
# Example: 4-bit x4 SR
python basicsr/train.py -opt options/train/train_srtquant_x4.yml --pruning 0.4\
 --force_yml bit=4 name=train_srtquant_x4_bit4
 ```
> Using slurm 
```bash
sbatch run_srtquant.sbatch --pruning 0.4 
 ```
>pruning denotes the desired pruning ratio. Adjust paths and parameters within run_srt.sh and run_srtquant.sbatch as needed.  
---

## Testing 

1. Ensure datasets and pretrained models are are available.
3. Run (choosing the <best_model.pth> from the logs):
> Without slurm 
   ```bash
   # Example: reproduce x2 bit 4 results from Table 2 
   python basicsr/test.py -opt options/test/test_srtquant_x2.yml --pruning 0.4\
          --force_yml bit=4 name=test_srtquant_x2_bit4 \
          path:pretrain_network_Q=experiments/train_srtquant_x2_bit4/models/<best_model.pth>
  ```
>With Slurm 
```bash
sbatch run_srtquant_test.sbatch --pruning 0.4
```
> Please update the relevant paths for the best model within run_srt_test.sh and run_srtquant_test.sbatch
---

## Statistics & significance testing

The `stats-files/` directory contains scripts to run our various statistical tests.

### 1) Normality comparison
> Are post-Hadamard tensors closer to Normal?
```bash
python stats-files/compare_normality.py
```
>(edit directory_path at bottom of the file, or import and call main("/path/..."))
> output is (Shapiro-W, K², AD, JB; deltas and significance).
### 2) Range reduction 
> analyze how Hadamard reduces ranges 
change the directory in the main function with path to weights_and_activs

```bash
python stats-files/range_reduction.py 
```
> output is deltas and test of significance

### 3) Concentration around 0 
>analyze how Hadamard concentrates values within the [-epsilon-epsilon] band
change the directory with path to weights_and_activs
other values of epsilon are supported 
```bash
python stats-files/epsilon_band.py path/to/weights_and_activs --eps 0.05 --by-type  
```
> output is deltas and test of significance

### 4) Results test  
> test of statistical significance on our main results 
```bash
python stats-files/results_wilcoxon.py 
```
> output is deltas and test of significance
---

## Acknowledgements

This repository is built on:

* [BasicSR](https://github.com/XPixelGroup/BasicSR)
* [2DQuant](https://github.com/Kai-Liu001/2DQuant)

---

## License

Apache 2.0 (see `LICENSE`).

---
