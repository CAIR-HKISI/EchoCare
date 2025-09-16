<div align="center">
<h1>EchoCare: A fully open and generalizable foundation model for ultrasound clinical applications</h1>

<a href="http://arxiv.org/abs/2509.11752"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/CAIR-HKISI'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://echocare.cares-copilot.com/'><img src='https://img.shields.io/badge/Homepage-EchoCare-green' alt='Homepage'></a>
</div>

This work presents **EchoCare**,  novel ultrasound foundation model for generalist clinical use, developed via self-supervised learning on our curated, publicly available, large-scale unlabeled dataset EchoCareData. EchoCareData comprises **4.5 million** ultrasound images, sourced from 23 clinical centers across 5 continents, and acquired using 38 distinct imaging devices, thus encompassing multi-center, multi-device, and multi-ethnic global cohorts.

![teaser](img/logo.png)

## Quick Start

- **EchoCareData**:
  - [Public Medical Dataset](#Public-Medical-Dataset)
  - [Other EchoCare Collection Projects](#Other-EchoCare-Collection-Projects)
  - [Dataset Platforms](#Dataset-Platforms)
- **Models**:
  - [Evaluation Benchmark](#Evaluation-Benchmark)
  - [Related SOTA Methods](#SOTA-Methods)
  - [Related Foundation Toolbox Projects](#Related-Foundation-Toolbox-Projects)
- **Results**:
  - [Node Segmentation](#Node-Segmentation)
  - [Vessel segmentation](#Vessel-segmentation)
  - [Organ segmentation](#Organ-segmentation)
  - [Node classification](#Node-classification)
  - [BI-BADS classification](#BI-BADS-classification)
  - [Lesion classification](#Lesion-classification)
  - [Organ detection](#Organ-detection)
  - [Landmark location](#Landmark-location)
  - [EF regression](#EF-regression)
  - [Image enhancement](#Image-enhancement)
  - [Report generation](#Report-generation)

## Public Medical Dataset

If you want to introduce or know more dataset, you are welcome to submit an issue or PR in this repo.

- [Imaging](#Imaging)
  - [Whole Body (10)](#whole-body)
  - [Head and Neck (55)](#head-and-neck)
  - [Chest (43)](#chest)
  - [Abdomen (53)](#abdomen)
  - [Heart (14)](#heart)
  - [Bones (15)](#bones)
  - [Endoscopy (33)](#endoscopy)
  - [Retina (52)](#retina)
  - [Skin (14)](#skin)
  - [Microscopic imaging (38)](#microscopic-imaging)
- [Imaging and Text (34)](#Image-text-dataset)
- [Text (18)](#Text-dataset)

## Dataset Platforms

You can search for more medical datasets not included in this project on these websites:

- [Grand Challenges](https://grand-challenge.org/challenges/): A platform for end-to-end development of machine learning solutions in biomedical imaging.
- [Kaggle](https://www.kaggle.com/datasets): One of the largest AI & ML community.
- [TCIA](https://www.cancerimagingarchive.net/access-data/): A service which de-identifies and hosts a large archive of medical images of cancer accessible for public download.
- [Synapse](https://www.synapse.org/): A platform for supporting scientific collaborations centered around shared biomedical data sets.
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/): The MSD challenge tests the generalisability of machine learning algorithms when applied to 10 different semantic segmentation tasks.
- [CodaLab](https://codalab.lisn.upsaclay.fr/): An open-source web-based platform that enables researchers, developers, and data scientists to collaborate, with the goal of advancing research fields where machine learning and advanced computation is used.
- [Tianchi (Chinese)](https://tianchi.aliyun.com/dataset): Tianchi is a developer competition platform under Alibaba Cloud.
- [OpenDataLab (Chinese)](https://opendatalab.com/): China Big Model Corpus Data Alliance open source data service designated platform, provides high-quality open data sets for large models.
