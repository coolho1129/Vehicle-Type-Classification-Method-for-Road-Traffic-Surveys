# Vehicle-Type-Classification-Method-for-Road-Traffic-Surveys

This repository contains the source code for a vehicle classification system, including initial vehicle classification, wheel detection, and trailer classification using machine learning models.

## Development Environment

- **Python Version:** 3.10.13
- **Operating System:** Ubuntu 20.04
- **GPU:** RTX A6000
- **CUDA Version:** 11.3

## Installation

Before running the scripts, set up the Python environment by creating a Conda environment using the provided `environment.yml` file:

```
conda env create -f environment.yml
```
Activate the environment:
```
conda activate vehicle_classify
```

## Usage  

### Initial Classifier Training

**Script**: `init_classifier_train.py`

**Purpose**: Trains the initial vehicle classifier using ResNet-50 and saves the model weights.

**Execution**: Navigate to vehicle_classifier and run:
```
python Init_classifier_train.py
```
**Outputs** : Model weights are saved in `./res_weight/resnet50/final`      
<br>

### Wheel Detection Training
**Script**: `wheel_detect_trainset.py`

**Purpose**: Detects wheels in images and saves the results along with wheel count and positions to CSV files.

**Execution**: Navigate to vehicle_classifier and run:
```
python wheel_detect_trainset.py
```
**Outputs**: Images and CSV files are saved in `./ml/train`.      
<br>


### Trailer Classifier Training

**Script**: `trailer_classifier_train.py`

**Purpose**: Trains classifiers to distinguish between full and semi trailers using extracted features.

**Execution**: Navigate to `vehicle_classifier` and run:
```
python trailer_classifier_train.py
```
**Outputs**: Best models are saved in `./ml/model`.      
<br> 


### Initial Classifier Prediction

**Script**: `init_classifier_predict.py`

**Purpose**: Classifies vehicles using the trained model and evaluates the accuracy.

**Execution**: Navigate to `vehicle_classifier` and run:
```
python Init_classifier_predict.py
```
**Outputs**: Predictions are saved in `./init_classification`.  
<br>


### Wheel Detection

**Script**: `wheel_detect.py`

**Purpose**: Detects wheels in new images and saves the detected images and data.

**Execution**: Navigate to `vehicle_classifier` and run:
```
python wheel_detect.py
```
**Outputs**: Images and CSV files are saved in `./output/pred`.  
<br>


### Trailer Classifier Prediction

**Script**: `trailer_classifier_predict.py`

**Purpose**: Classifies trailers using the trained models and saves the results.

**Execution**: Navigate to `vehicle_classifier` and run:
```
python trailer_classifier_predict.py
```
**Outputs**: Classification results are saved in `./result`.
