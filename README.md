<div align="center">
<p align="center">
  <h2>The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation</h2>
  <a href="https://arxiv.org/abs/2503.10636">Paper</a> | <a href="https://hkchengrex.github.io/C2OT">Webpage</a> | <a href="https://colab.research.google.com/drive/1uhYPqnGlPoMTEqEgzpPvFQEcnr0faSBA?usp=sharing">Colab</a> 
</p>
<p>
<a href="https://hkchengrex.github.io/">Ho Kei Cheng</a> and 
<a href="https://www.alexander-schwing.de/">Alexander Schwing</a>
<br>
University of Illinois Urbana-Champaign
</p>
</div>

![8GtoMoons](https://imgur.com/bcmTUiE.png)

## High-Level Summary

C<sup>2</sup>OT is an algorithm for computing prior-to-data couplings for flow-matching-based generative models during training.
Our goal is to achieve straighter flows, enabled by optimal transport (OT) couplings, while mitigating the test-time degradation that OT encounters in the conditional setting (see figure above).
The key idea is that OT samples from a condition-skewed prior distribution at test time, whereas C<sup>2</sup>OT unskews the prior by incorporating a condition-dependent term into the OT cost.


## Installation

We have only tested this on Ubuntu.

### Prerequisites

We recommend using a [miniforge](https://github.com/conda-forge/miniforge) environment.

- Python 3.9+
- PyTorch 2.5.1+ and corresponding torchvision/torchaudio (pick your CUDA version https://pytorch.org/, pip install recommended)

**1. Install prerequisite if not yet met:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

(Or any other CUDA versions that your GPUs/driver support)

<!-- ```
conda install -c conda-forge 'ffmpeg<7
```
(Optional, if you use miniforge and don't already have the appropriate ffmpeg) -->

**2. Clone our repository:**

```bash
git clone https://github.com/hkchengrex/C2OT.git
```

**3. Install with pip (install pytorch first before attempting this!):**

```bash
cd C2OT
pip install -e .
```

(If you encounter the `File "setup.py" not found` error, upgrade your pip with pip install --upgrade pip)

## Demo

After installation, you can run our demo notebook at [moons.ipynb](moons.ipynb). 
This demo contains pretty much everything you need to know about C<sup>2</sup>OT.
The rest of this repository is for reproducing our results in CIFAR-10 and ImageNet-32.

You can also run this demo on [Colab](https://colab.research.google.com/drive/1uhYPqnGlPoMTEqEgzpPvFQEcnr0faSBA?usp=sharing) without a local installation.


## CIFAR-10

### Data Preparation

You don't need to prepare the dataset manually. The code should download the dataset automatically. You can change the dataset path in `config/cifar.yaml` if you want to.

### Training

To train, use `torchrun` (two GPUs, ~10G memory usage each):

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc-per-node=2 train.py --config-name=cifar  exp_id=<some unqiue experiment identifier> fm_type=<fm/ot/c2ot>
```

The trained model and the logs will be saved in `outputs/cifar/<exp_id>`.
The code will also automatic evaluate the model after training.

### Evaluation

To evaluate, 
```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc-per-node=2 train.py --config-name=cifar  exp_id=<some unqiue experiment identifier> checkpoint=<path to the pretrained checkpoint>
```

The generated samples (as zip files) will be saved in `outputs/cifar/<exp_id>`.


## ImageNet-32

### Data Preparation


#### 1. Download the face-blurred ImageNet train and validation set from https://image-net.org/download-images.php
#### 2. Downsample the data to 32x32 using https://github.com/PatrykChrabaszcz/Imagenet32_Scripts, e.g., 
```bash
python Imagenet32_Scripts/image_resizer_imagent.py -i ../imagenet/train_blurred -o ../imagenet/train_blurred_32 -s 32 -a box  -r -j 10 
```
Replace the path to the dataset with your own path. Repeat this for the validation set.

#### 3. Precompute FID statistics on the validation set using
```bash
python -c "from cleanfid import fid; fid.make_custom_stats('imagenet32_val', '../imagenet/train_blurred_32/box', mode='legacy_tensorflow', num_workers=16, batch_size=256)"
```
Replace the path with the path to your downsampled validation set. 

#### 4. Download the captions
Download the captions from https://huggingface.co/datasets/visual-layer/imagenet-1k-vl-enriched. You might have to download the entire dataset and only keep `train_captions.json` and `val_captions.json`.

#### 5. Precompute the CLIP features (to speed up training)
```bash
python scripts/extract_clip_captions.py --imagenet_path ../imagenet/train_blurred_32/box --caption_path ../imagenet-1k-vl-enriched/train_captions.json --output_path ../imagenet/train_clip_captions.pth
```
Replace the path with the path to your downsampled training set and the captions.
Repeat this for the validation set.

#### 6. Build TensorDict for Imagenet (to speed up data loading)
```bash
python scripts/make_memmap.py --input_dir ../imagenet/train_blurred_32/box --output_dir ../imagenet/train_blurred_32/train_memmap --clip_features ../imagenet/train_clip_captions.pth
```
Replace the path with the path to your downsampled training set and the captions.
You do not need to do this for the validation set.

#### 7. Update `config/imagenet32.yaml`:
- `data_path`: path to `train_memmap`
- `val_clip_path`: path to `val_clip_captions.pth`


### Training
To train, use `torchrun` (four GPUs, ~12G memory usage each):

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc-per-node=4 train.py --config-name=imagenet32 exp_id=<some unqiue experiment identifier> fm_type=<fm/ot/c2ot>
```

The trained model and the logs will be saved in `outputs/imagenet32/<exp_id>`.
The code will also automatic evaluate the model after training.

### Evaluation
To evaluate, 
```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc-per-node=4 train.py --config-name=imagenet32 exp_id=<some unqiue experiment identifier> checkpoint=<path to the pretrained checkpoint>
```
The generated samples (as zip files) will be saved in `outputs/imagenet32/<exp_id>`.


## Citation

```bibtex
@inproceedings{cheng2025curse,
  title={The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation},
  author={Cheng, Ho Kei and Schwing, Alexander},
  booktitle={arXiv},
  year={2025}
}
```

## Acknowledgement

Many thanks to:
- [torchcfm](https://github.com/atong01/conditional-flow-matching)