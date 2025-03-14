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

C<sup>2</sup>OT is an algorithm for computing prior-to-data couplings for flow-matching-based generative models at training time. 
Our goal is to obtain straighter flows granted by optimal transport (OT) couplings, while preventing test-time degradation that OT has in the conditional setting (see figure above).
The idea is that OT samples from a condition-skewed prior distribution while C<sup>2</sup>OT *unskews* the prior distribution by adding a condition-dependent term to the OT cost.


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

(If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip)

## Demo

After installation, you can run our demo notebook at [moons.ipynb](moons.ipynb). More details to be added. 
You can also run this demo on [Colab](https://colab.research.google.com/drive/1uhYPqnGlPoMTEqEgzpPvFQEcnr0faSBA?usp=sharing) without a local installation.

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