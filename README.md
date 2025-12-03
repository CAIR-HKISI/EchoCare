<div align="center">
<h1>EchoCare: A fully open and generalizable foundation model for ultrasound clinical applications</h1>

<a href="http://arxiv.org/abs/2509.11752"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/CAIR-HKISI'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://echocare.cares-copilot.com/'><img src='https://img.shields.io/badge/Homepage-EchoCare-green' alt='Homepage'></a>
</div>

This work presents **EchoCare**,  novel ultrasound foundation model for generalist clinical use, developed via self-supervised learning on our curated, publicly available, large-scale unlabeled dataset EchoAtlas. EchoAtlas comprises **4.5 million** ultrasound images, sourced from 23 clinical centers across 5 continents, and acquired using 38 distinct imaging devices, thus encompassing multi-center, multi-device, and multi-ethnic global cohorts.

![EchoCare](img/logo.png)

## Quick Start

- **EchoAtlas**:
  - [Public Medical Dataset](#Public-Medical-Dataset)
  - [Dataset Platforms](#Dataset-Platforms)
- **Models**:
  - [Usage](#Usage)
  - [Evaluation Benchmark](#Evaluation-Benchmark)
  - [Related SOTA Methods](#SOTA-Methods)
  - [Related Foundation Toolbox Projects](#Related-Foundation-Toolbox-Projects)
- **Results**:
  - [Node classification](#Node-classification)
  - [BI-BADS classification](#BI-BADS-classification)
  - [Lesion classification](#Lesion-classification)
  - [Node Segmentation](#Node-Segmentation)
  - [Vessel segmentation](#Vessel-segmentation)
  - [Organ segmentation](#Organ-segmentation)
  - [Organ detection](#Organ-detection)
  - [Landmark location](#Landmark-location)
  - [EF regression](#EF-regression)
  - [Image enhancement](#Image-enhancement)
  - [Report generation](#Report-generation)

## Public Medical Dataset

EchoAtlas covers 9 major regions and 52 anatomical organs of the human body, supporting models pretrained on it to generalize effectively across comprehensive whole-body ultrasound clinical application.

- [Abdomen](#Abdomen)
- [Back](#Back)  
- [Fetus](#Fetus)  
- [Head&Neck](#Head&Neck)  
- [Lower limb](#Lower-limb)  
- [Other](#Other)  
- [Pelvis](#Pelvis)  
- [Thorax](#Thorax)  
- [Upper limb](#Upper-limb)  

## Dataset Platforms

Our data curation process commenced with a systematic search of open academic repositories：

- [Figshare](https://figshare.com/): An online repository where researchers can share, manage, and showcase research outputs with DOIs for citation.  
- [Github](https://github.com/): The world's leading platform for hosting and collaborating on code projects.  
- [Grand-challenge](https://grand-challenge.org/): A platform for hosting medical imaging challenges and datasets.  
- [Kaggle](https://www.kaggle.com/datasets): One of the largest AI & ML community.  
- [Mendeley](https://www.mendeley.com/): A reference manager and academic social network for researchers.  
- [Zenodo](https://zenodo.org/): An open-access repository for research outputs and datasets.  


## Usage

Before training or inference, load the [pre-trained weights](https://cashkisi-my.sharepoint.com/:u:/g/personal/cares-copilot_cair-cas_org_hk/IQBgK6rK8TAtQq8IjADsgp52AbmyC03ubimwqr3qh8ZH6DI?e=ABYQzg). This gives you a well-initialized model that can be fine-tuned for your own dataset or task.

```
import torch
import argparse
from monai.networks.nets.swin_unetr import SwinTransformer

parser = argparse.ArgumentParser(description="Swin Transformer")
parser.add_argument("--feature_size", default=128, type=int, help="feature size")
parser.add_argument("--in_channels", default=3, type=int, help="number of input channels")
parser.add_argument("--pretrained_checkpoint", default=None, help="encoder pretrained checkpoint")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
args = parser.parse_args()

encoder = SwinTransformer(
    in_chans=args.in_channels,
    embed_dim=args.feature_size,
    window_size=[8] * 2,
    patch_size=[2] * 2,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    mlp_ratio=4.0,
    qkv_bias=True,
    use_checkpoint=args.use_checkpoint,
    spatial_dims=2,
    use_v2=True)

if args.pretrained_checkpoint is not None:
    model_dict = torch.load(args.pretrained_checkpoint, map_location=torch.device('cpu'))
    state_dict = model_dict
    state_dict.pop('mask_token')
    encoder.load_state_dict(state_dict, strict=True)
    print("Using pretrained self-supervised Swin Transformer backbone weights !")

# Test case: forward pass with dummy input
# Expected output feature map shapes:
# [1, 128, 128, 128], [1, 256, 64, 64], [1, 512, 32, 32], [1, 1024, 16, 16], [1, 2048, 8, 8]
x = torch.rand(1, 3, 256, 256)
x_outs = encoder(x)
print([x_out.shape for x_out in x_outs])
```


## Acknowledgement
We thank [MONAI](https://github.com/Project-MONAI/research-contributions) for part of their codes.
