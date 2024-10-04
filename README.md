# SAR2Optical

## Overview

This project implements a deep learning model for translating Synthetic Aperture Radar (SAR) images to optical images using the Pix2Pix framework. The goal is to improve the interpretability of SAR data by converting it into a format that is more accessible and useful for various applications, including remote sensing, environmental monitoring, and urban planning.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [License](#license)
- [Acknowledgments](#acknowledgments)


## Dataset

For this project, paired SAR and optical (RGB) images from the Sentinel‑1 and Sentinel‑2 satellites are used to train the models. The dataset source is [Sentinel-1&2 Image Pairs, Michael Schmitt, Technical University of Munich (TUM)](https://mediatum.ub.tum.de/1436631). The dataset is downloaded from Kaggle at [Sentinel-1&2 Image Pairs (SAR & Optical)](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain), uploaded by [Paritosh Tiwari (@requiemonk)](https://www.kaggle.com/requiemonk).

The dataset is divided into three splits: training, validation, and testing. We randomly sampled a total of 1,000 image pairs for testing, with 250 pairs from each of the four classes. Similarly, another 1,000 pairs were allocated for validation, also with same distribution. The remaining 14,000 image pairs were designated for training. You can find the IDs for each split in the [split.txt](/data/split.txt).

## Installation

1. Clone the repository:

   ```bash
   https://github.com/yuuIind/SAR2Optical.git
   cd SAR2Optical
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Prepare data
Before training, you need to prepare the dataset. Check [Dataset](#dataset) section for more information on how to find the dataset.

### Train
To train the model, run the following command:
   ```bash
   Command here
   ```

### Evaluate
To evaluate the model’s performance, you can use metrics such as PSNR and SSIM. Example command:
   ```bash
   Command here
   ```

### Inference
Once the model is trained, you can use it to translate SAR images to optical images. Optionally, you can use the provided checkpoints:
   ```bash
   Command here
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Sentinel-1&2 Image Pairs, Michael Schmitt, Technical University of Munich (TUM)](https://mediatum.ub.tum.de/1436631) 
```
@misc{1436631,
	author = {Schmitt, Michael},
	title = {{SEN1-2}},
	year = {2018},
	type = {Dataset},
	abstract = {SEN1-2 is a dataset consisting of 282,384 pairs of corresponding
synthetic aperture radar and optical image patches acquired by the Sentinel-1 and Sentinel-2 remote sensing satellites, respectively.},
	keywords = {Remote sensing, deep learning, data fusion, synthetic aperture radar imagery, optical imagery},
	doi = {},
	note = {},
}
```
- [Image-to-Image Translation Using Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1):
```
@article{pix2pix2017,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={CVPR},
  year={2017}
}
```
