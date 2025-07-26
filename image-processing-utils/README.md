# Image Processing Toolkit

This repository offers a collection of image processing utilities developed in Python. These tools facilitate color-space conversions, image augmentation techniques, and the creation of image pyramids for various applications in computer vision and image analysis.

---

##  Features

###  HSV Color Space Manipulation (`hsv_color_editor.py`)

Easily adjust the hue, saturation, and brightness of images using HSV color space.

#### How to use:

```bash
python hsv_color_editor.py <image_path> <hue_shift> <saturation_scale> <value_scale>
```

* `<image_path>`: Path to input image (e.g., `image.jpg`).
* `<hue_shift>`: Hue adjustment in degrees (e.g., `30`).
* `<saturation_scale>`: Saturation adjustment factor (e.g., `0.8`).
* `<value_scale>`: Brightness adjustment factor (e.g., `0.5`).

Example:

```bash
python hsv_color_editor.py sunset.jpg 45 1.2 0.9
```

---

###  Image Augmentation (`image_augmentation.py`)

Provides essential functions for augmenting image data:

* **Random Square Crop**: Extract random square patches.
* **Resizing**: Quickly resize images using nearest neighbor interpolation.
* **HSV Jittering**: Introduce random variations in image colors for robust training data.

You can integrate these utilities into your image preprocessing pipelines easily.

---

###  Image Pyramid Generator (`image_pyramid_generator.py`)

Automatically generate scaled-down images for multi-resolution analysis.

#### How to use:

```bash
python image_pyramid_generator.py <image_path>
```

Example Output:

```
image_2x.jpg
image_4x.jpg
image_8x.jpg
```

This functionality is particularly useful in tasks like image blending, object detection, and scale-space feature extraction.

---

##  Repository Structure

```
image-processing-toolkit/
├── hsv_color_editor.py
├── image_augmentation.py
├── image_pyramid_generator.py
├── sample_images/       # Optional: Store demo images here
└── README.md
```

---

##  Installation

Ensure you have Python 3.8+ and install dependencies via:

```bash
pip install numpy opencv-python pillow
```

---

## 🛠️ Potential Improvements

* Batch processing for multiple images simultaneously.
* CLI interface using `argparse` for enhanced usability.
* Integration with GUI tools like Streamlit for interactive demonstrations.

---

## 📖 Background

Initially built to explore and implement fundamental image processing techniques, these tools have been refined for practical use in diverse image analysis projects.

Feel free to explore, adapt, and contribute!

---


