# Feature Extraction and Image Classification Toolkit

This repository presents a comprehensive implementation of classical feature extraction and classification techniques for images using Python. The focus is on comparing the effectiveness of Histogram of Oriented Gradients (HOG) and Scale-Invariant Feature Transform (SIFT) in the context of image classification using Support Vector Machines (SVM).

---

## 📦 Project Modules

### 🔍 `keypoint_matcher.py`

Detects and visualizes keypoints between a pair of images using SIFT descriptors.

* Extracts rotation and scale-invariant features.
* Matches descriptors between images.
* Visualizes connections between matched keypoints.

**Usage:**

```bash
python keypoint_matcher.py
```

Ensure `Rainier1.png` and `Rainier2.png` are in the same directory.

---

### 📂 `cifar_loader.py`

Loads and prepares the CIFAR-10 dataset.

* Saves the dataset as `cifar10.npz`.

**Usage:**

```bash
python cifar_loader.py
```

Must be run before feature extraction or evaluation scripts.

---

### ⚙️ `extract_features.py`

Extracts features using both HOG and SIFT:

* **`extract_hog(images)`**: Converts images to grayscale and extracts HOG features.
* **`extract_sift(images, labels)`**: Uses SIFT to extract keypoints and descriptors.
* **`create_bovw(descriptors_list)`**: Builds Bag-of-Visual-Words vocabulary using K-means.

**Output:**

* `hog_features.npz`
* `sift_features.npz`

**Usage:**

```bash
python extract_features.py
```

---

### 📊 `classify_with_hog.py`

Trains an SVM classifier using HOG features and evaluates performance.

* Loads `hog_features.npz`
* Prints classification statistics and accuracy.

**Usage:**

```bash
python classify_with_hog.py
```

---

### 📊 `classify_with_sift.py`

Trains an SVM classifier using SIFT + BoVW features.

* Loads `sift_features.npz`
* Prints classification accuracy and insights.

**Usage:**

```bash
python classify_with_sift.py
```

---

## 📈 Performance Summary

| Method | Accuracy | Features/Image | Training Samples |
| ------ | -------- | -------------- | ---------------- |
| HOG    | 37.23%   | 324            | 16,000           |
| SIFT   | 10.09%   | 500            | 15,933           |

HOG outperformed SIFT on the CIFAR-10 dataset despite its simpler design and lower feature dimensionality.

---

## 🧠 Insights

* HOG is more effective for small-scale image classification tasks due to its grid-based, gradient-based descriptors.
* SIFT’s high dimensionality and scale-space complexity may reduce classification performance when used with simple BoVW pipelines.
* Bag-of-Words quantization may lead to a loss of discriminative power.

---

## 🛠️ Requirements

* Python 3.8+
* `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `tqdm`

Install via:

```bash
pip install numpy opencv-python scikit-image scikit-learn tqdm
```

---

## 📂 Repository Structure

```
image-feature-extraction/
├── keypoint_matcher.py
├── cifar_loader.py
├── extract_features.py
├── classify_with_hog.py
├── classify_with_sift.py
├── Rainier1.png
├── Rainier2.png
├── sample_outputs/       # Optional folder for visuals and results
└── README.md
```

---

## 📝 License

MIT License
