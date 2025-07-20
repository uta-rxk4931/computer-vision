# Computer Vision Projects 

This repository contains multiple computer vision projects developed for the CSE 4310 course at the University of Texas at Arlington. Each project focuses on a unique aspect of image processing, feature extraction, motion tracking, or deep learning.

---

## 1. Motion Tracking with Kalman Filters

Detects and tracks multiple moving objects in a video using frame differencing and Kalman Filters.

- Implements motion detection using temporal frame differences.
- Uses morphological operations and connected components for blob detection.
- Applies Kalman filtering to predict and track object movement.
- GUI built with PySide6 to visualize motion, bounding boxes, and trails.

Files:  
`motion_detector.py`, `qtdemo.py`, `Demo.mp4`, `environment.yml`

---

## 2. Image Processing Toolkit

Provides tools for common image processing tasks and data augmentation.

- HSV color adjustment: Modify hue, saturation, and brightness.
- Random crop and resizing: Useful for training image models.
- Image pyramid generation: For multi-scale processing or blending.

Files:  
`hsv_color_editor.py`, `image_augmentation.py`, `image_pyramid_generator.py`

---

## 3. Feature Extraction and Image Classification

Compares classical image features using machine learning models on CIFAR-10.

- Uses Histogram of Oriented Gradients (HOG) and Scale-Invariant Feature Transform (SIFT).
- Builds a Bag-of-Visual-Words vocabulary from SIFT descriptors.
- Trains Support Vector Machines (SVMs) to classify images based on extracted features.

Files:  
`cifar_loader.py`, `keypoint_matcher.py`, `extract_features.py`, `classify_with_hog.py`, `classify_with_sift.py`

---

## 4. CNN Architectures and Transfer Learning

Explores deep learning approaches using convolutional neural networks (CNNs) on the Imagenette dataset.

- Builds a basic CNN with pooling layers.
- Implements an All Convolutional Network (All-CNN) without pooling.
- Adds regularization techniques such as dropout and weight decay.
- Applies transfer learning with pretrained weights to fine-tune the model.

Files:  
`Basic_CNN.ipynb`, `ALLConv.ipynb`, `Regularization.ipynb`, `TransferLearning.ipynb`, `Model Weights/`, `Report.pdf`

---

## Setup and Dependencies

Each project includes its own instructions. Some require Jupyter notebooks, others are run as Python scripts.

Install general dependencies:

```bash
pip install numpy opencv-python scikit-image scikit-learn torch torchvision pytorch-lightning pyside6 tqdm
```

Use the provided `environment.yml` file for motion tracking to create a Conda environment:

```bash
conda env create -f environment.yml
conda activate cse4310tracking
```

---

## Author

Rency Ajit Kansagra  
University of Texas at Arlington
