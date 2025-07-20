import numpy as np
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import dilation


class MotionDetector:
    
    def __init__(
        self,
        tau=0.05,  # τ - motion threshold
        alpha=3,  # α - frame hysteresis
        delta=50,  # δ - distance threshold
        skip_frames=0,  # s - frames to skip
        max_objects=5,  # N - max objects to track
        dilation_size=9,  # size of dilation kernel
        min_area=100,  # minimum blob area
    ):

        # Detection parameters
        self.tau = tau
        self.dilation_size = dilation_size
        self.min_area = min_area

        # Tracking parameters
        self.alpha = alpha
        self.delta = delta
        self.skip_frames = skip_frames
        self.max_objects = max_objects

        # State variables
        self.prev_frame = None
        self.prev_prev_frame = None
        self.tracked_objects = []  # List of KalmanFilter instances
        self.object_status = []  # List of (active_frames, inactive_frames)
        self.candidates = []  # List of (centroid, frame_count, active_count, region_idx)
        self.frame_count = 0
        self.frames_to_skip = 0

    def update(self, current_frame):
    
        self.frame_count += 1

        # Convert frame to grayscale if needed
        cframe = current_frame if current_frame.ndim == 2 else rgb2gray(current_frame)

        # Initialize frame history if needed
        if self.prev_prev_frame is None:
            self.prev_prev_frame = cframe
            return []

        if self.prev_frame is None:
            self.prev_frame = cframe
            return []

        # Handle frame skipping
        if self.frames_to_skip > 0:
            self.frames_to_skip -= 1
            self._predict_all()  # Predict object positions even when skipping frames
            return self._get_active_objects()

        # Calculate frame differences (motion)
        diff1 = np.abs(cframe - self.prev_frame)  # Current vs. previous
        diff2 = np.abs(self.prev_frame - self.prev_prev_frame)  # Previous vs. prev-prev
        motion_frame = np.minimum(diff1, diff2)  # Pixel-wise minimum

        # Threshold to remove noise
        thresh_frame = motion_frame > self.tau

        # Dilate to connect nearby motion pixels
        dilated_frame = dilation(
            thresh_frame, np.ones((self.dilation_size, self.dilation_size))
        )

        # Find connected regions (blobs)
        label_frame = label(dilated_frame)
        regions = [r for r in regionprops(label_frame) if r.area >= self.min_area]

        # Update frame history
        self.prev_prev_frame = self.prev_frame
        self.prev_frame = cframe
        self.frames_to_skip = self.skip_frames  # Reset skip counter

        # Convert region properties to detections (centroids)
        detections = [r.centroid[::-1] for r in regions]  # Convert (row, col) to (x, y)

        # Update object tracking
        self._update_tracking(detections, regions)

        # Return the list of active tracked objects
        return self._get_active_objects()

    def _predict_all(self):
     
        for obj in self.tracked_objects:
            obj.predict()

    def _update_tracking(self, detections, regions):
        # Predict the current state for all tracked objects
        predictions = [obj.predict() for obj in self.tracked_objects]

        # Associate detections with tracked objects
        used_detections = set()
        for i, (obj, status) in enumerate(zip(self.tracked_objects, self.object_status)):
            active, inactive = status

            # Find the closest detection to the predicted object location
            min_dist = float("inf")
            best_idx = -1
            for j, centroid in enumerate(detections):
                if j in used_detections:
                    continue
                dist = np.linalg.norm(np.array(centroid) - np.array(predictions[i]))
                if dist < min_dist and dist < self.delta:
                    min_dist = dist
                    best_idx = j

            if best_idx != -1:
                # Update the Kalman filter with the associated detection
                obj.update(detections[best_idx])
                obj.bbox = regions[best_idx].bbox
                used_detections.add(best_idx)
                self.object_status[i] = (active + 1, 0)  # Increment active, reset inactive
            else:
                # Increment inactive frames
                self.object_status[i] = (active, inactive + 1)

        # Update or create candidate objects for unassociated detections
        new_candidates = []
        for j, centroid in enumerate(detections):
            if j in used_detections:
                continue
            # Check if centroid matches an existing candidate
            matched = False
            for cand in self.candidates:
                cand_centroid, cand_frame, cand_active, _ = cand
                if np.linalg.norm(np.array(centroid) - np.array(cand_centroid)) < self.delta:
                    cand[0] = centroid  # Update position
                    cand[2] += 1  # Increment active count
                    cand[3] = j  # Update region index
                    matched = True
                    if cand[2] >= self.alpha and len(self.tracked_objects) < self.max_objects:
                        # Promote candidate to tracked object
                        new_obj = KalmanFilter(centroid)
                        new_obj.bbox = regions[j].bbox
                        self.tracked_objects.append(new_obj)
                        self.object_status.append((cand[2], 0))
                    else:
                        new_candidates.append(cand)
                    break
            if not matched:
                # Add new candidate
                new_candidates.append([centroid, self.frame_count, 1, j])

        # Remove stale candidates
        self.candidates = [c for c in new_candidates if self.frame_count - c[1] < self.alpha]

        # Remove inactive objects
        to_remove = [i for i, (_, inactive) in enumerate(self.object_status) if inactive >= self.alpha]
        for i in sorted(to_remove, reverse=True):
            self.tracked_objects.pop(i)
            self.object_status.pop(i)

    def _get_active_objects(self):
       
        active_objects = []
        for obj, (active, _) in zip(self.tracked_objects, self.object_status):
            if active >= self.alpha:  # Only include objects active for alpha frames
                obj_data = {
                    "position": obj.state[:2],
                    "velocity": obj.state[2:],
                    "history": obj.history.copy(),
                }
                if hasattr(obj, "bbox"):
                    obj_data["bbox"] = obj.bbox
                active_objects.append(obj_data)
        return active_objects

    def reset(self):
        
        self.prev_frame = None
        self.prev_prev_frame = None
        self.tracked_objects = []
        self.object_status = []
        self.candidates = []
        self.frame_count = 0
        self.frames_to_skip = 0


class KalmanFilter:

    def __init__(self, initial_pos, delta_t=1.0):
    
        # State: [x, y, vx, vy] (position and velocity)
        self.state = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=float)
        self.covariance = np.eye(4) * 10.0  # Initial state covariance

        # State transition matrix
        self.F = np.array(
            [
                [1, 0, delta_t, 0],
                [0, 1, 0, delta_t],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        # Process noise covariance
        self.Q = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 10, 0],
                [0, 0, 0, 10],
            ],
            dtype=float,
        ) * 0.1

        # Measurement matrix (measures position only)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

        # Measurement noise covariance
        self.R = np.eye(2) * 5.0

        # Tracking history
        self.history = [initial_pos]
        self.bbox = None  # Bounding box

    def predict(self):
    
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state[:2]

    def update(self, measurement):
    
        z = np.array(measurement, dtype=float)
        y = z - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
        self.history.append(self.state[:2])
        return self.state[:2]