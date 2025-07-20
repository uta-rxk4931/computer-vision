# Motion Tracking 

## DEMO
Watch the demo video: Demo.mp4 (same folder)

## Code Functionality

### Motion Detection Based on Frame Differences
The `MotionDetector` class (`motion_detector.py`) detects motion by computing two frame differences:
- Between the current frame and the previous frame (`diff1`).
- Between the previous frame and the frame before that (`diff2`).
The minimum of these two differences is thresholded using a hyperparameter `τ` (tau) to reduce noise.

### Noise Filtering and Dilation
After thresholding, the detected motion pixels are dilated using a 9×9 window (`self.dilation_size`), implemented via `skimage.morphology.dilation`, connecting nearby blobs. 

### Connected Components and Candidate Extraction
The blobs are labeled using `skimage.measure.label`, and regions with sufficient area (`min_area`) are kept. Their centroids are computed with `regionprops`, allowing each motion blob to be treated as an object candidate.

### Kalman Filter Object Tracking
Each tracked object is managed by an instance of the `KalmanFilter` class:
- Predicts the object's next position (`predict()`).
- Updates the position based on new measurements (`update()`).
Each filter maintains a history of the object's past locations for trail visualization.

Objects are added after being consistently active for at least `α` frames and removed if inactive for `α` frames — following the assignment's tracking logic exactly.

### Tracking Multiple Objects and Frame Skipping
The code supports tracking up to `N` objects (`max_objects`) and can skip `s` frames (`skip_frames`) between detections while still predicting object positions.

### GUI with Frame Navigation and Visualization
The `qtdemo.py` file provides a full PySide6 GUI:
- Displays the video frame-by-frame.
- Allows frame-by-frame navigation with "Previous" and "Next" buttons.
- Jump 60 frames forward/backward.
- Optionally shows bounding boxes and movement trails for tracked objects.
The UI reinitializes tracking properly when jumping frames.

### Trail Drawing and Bounding Boxes
For visualization, each active tracked object draws:
- Bounding boxes around the motion regions.
- Trails showing the movement history over time.

NOTE: My implementation directly follows the professor’s reference code for both motion detection (frame_diff) and object tracking (kalman_filter) using Kalman Filters, matching the prescribed prediction-update cycle and motion blob detection logic.

---

## What's Required to Run the Code

1. **Conda Environment**  
   Use the provided `environment.yml` to set up the required environment:
   ```bash
   conda env create -f environment.yml
   conda activate cse4310tracking

2. **Required Packages**
- pyside6
- numpy
- scikit-video
- scikit-image

3. **Files Needed**
- `motion_detector.py`
- `qtdemo.py`
- A video file (e.g., `east_parking_reduced_size.mp4`)

4. **Running the Program**
```bash
python qtdemo.py east_parking_reduced_size.mp4

## Author

Rency Ajit Kansagra 

University of Texas at Arlington  
