# Vehicle Type Classification Method for Road Traffic Surveys

This repository provides the implementation of a vehicle type classification system for road traffic surveys. The system integrates initial vehicle classification, wheel detection, and trailer classification using deep learning and machine learning techniques.

## Publication

**Vehicle Type Classification Method for Road Traffic Surveys**

Mi-Seon Kang, Chan-Ho Kim, Pyong-Kun Kim

*IEMEK Journal of Embedded Systems and Applications*, Vol. 19, No. 5, pp. 227–234, 2024.

[**DOI**](https://doi.org/10.14372/IEMEK.2024.19.5.227) | [KCI](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003132180)

## Development Environment

- **OS:** Ubuntu 20.04
- **Python:** 3.10.13
- **GPU:** NVIDIA RTX A6000
- **CUDA:** 11.3

## Installation

Create the Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate vehicle_classify
```

Navigate to the source directory before running the scripts:

```bash
cd vehicle_classifier
```

## Training

### 1. Initial Vehicle Classifier

Train a ResNet-50-based classifier for initial vehicle type classification.

```bash
python init_classifier_train.py
```

**Output:** Trained model weights are saved in `./res_weight/resnet50/final`.

### 2. Wheel Detection

Detect vehicle wheels and extract wheel counts and positions for the training dataset.

```bash
python wheel_detect_trainset.py
```

**Output:** Processed images and CSV files are saved in `./ml/train`.

### 3. Trailer Classifier

Train machine learning classifiers to distinguish between full trailers and semi-trailers using extracted features.

```bash
python trailer_classifier_train.py
```

**Output:** Best-performing models are saved in `./ml/model`.

## Inference

### 1. Initial Vehicle Classification

Perform vehicle type classification using the trained ResNet-50 model and evaluate classification accuracy.

```bash
python init_classifier_predict.py
```

**Output:** Classification predictions are saved in `./init_classification`.

### 2. Wheel Detection

Detect wheels in input vehicle images and extract wheel-related information.

```bash
python wheel_detect.py
```

**Output:** Detection results, including annotated images and CSV files, are saved in `./output/pred`.

### 3. Trailer Classification

Classify trailers as full trailers or semi-trailers using the trained machine learning models.

```bash
python trailer_classifier_predict.py
```

**Output:** Classification results are saved in `./result`.
