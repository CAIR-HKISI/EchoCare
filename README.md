<div align="center">
<h1>EchoCare: A fully open and generalizable foundation model for ultrasound clinical applications</h1>

<a href="http://arxiv.org/abs/2509.11752"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/CAIR-HKISI'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://echocare.cares-copilot.com/'><img src='https://img.shields.io/badge/Homepage-EchoCare-green' alt='Homepage'></a>
</div>

This work presents **EchoCare**,  novel ultrasound foundation model for generalist clinical use, developed via self-supervised learning on our curated, publicly available, large-scale unlabeled dataset EchoCareData. EchoCareData comprises **4.5 million** ultrasound images, sourced from 23 clinical centers across 5 continents, and acquired using 38 distinct imaging devices, thus encompassing multi-center, multi-device, and multi-ethnic global cohorts.

![EchoCare](img/logo.png)

## Quick Start

- **EchoCareData**:
  - [Public Medical Dataset](#Public-Medical-Dataset)
  - [Dataset Platforms](#Dataset-Platforms)
- **Models**:
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

EchoCareData covers 9 major regions and 52 anatomical organs of the human body, supporting models pretrained on it to generalize effectively across comprehensive whole-body ultrasound clinical application.

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
