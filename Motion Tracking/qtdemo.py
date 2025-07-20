import sys
import argparse
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
from skvideo.io import vread
from skimage.color import rgb2gray
from motion_detector import MotionDetector


np.float = np.float64
np.int = np.int_


class QtDemo(QtWidgets.QWidget):

    def __init__(self, frames):
        
        super().__init__()

        self.frames = frames
        self.current_frame = 0
        self.processed_frames = (
            {}
        )  
        self.detector = MotionDetector(
            tau=0.05,  # Motion threshold
            alpha=5,  # Frame hysteresis
            delta=50,  # Distance threshold
            skip_frames=0,  # Frames to skip
            max_objects=5,  # Max objects to track
            dilation_size=9,  # Dilation kernel size
            min_area=100,  # Minimum blob area
        )

        self.setup_ui()  
        self.update_tracking()  

    def setup_ui(self):

        self.setWindowTitle("Object Tracking Demo")

        # Image display area
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)

        # Navigation buttons
        self.prev_button = QtWidgets.QPushButton("Previous Frame")
        self.next_button = QtWidgets.QPushButton("Next Frame")
        self.jump_prev_button = QtWidgets.QPushButton("<< -60 Frames")
        self.jump_next_button = QtWidgets.QPushButton("+60 Frames >>")

        # Frame slider for manual frame selection
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, len(self.frames) - 1)
        self.frame_slider.setTickInterval(1)

        # Checkboxes to control visualization options
        self.show_trails = QtWidgets.QCheckBox("Show Movement Trails")
        self.show_trails.setChecked(True)
        self.show_boxes = QtWidgets.QCheckBox("Show Bounding Boxes")
        self.show_boxes.setChecked(True)

        # Layout organization
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.jump_prev_button)
        button_row.addWidget(self.prev_button)
        button_row.addWidget(self.next_button)
        button_row.addWidget(self.jump_next_button)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.addWidget(self.show_trails)
        controls_row.addWidget(self.show_boxes)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.img_label)
        main_layout.addLayout(button_row)
        main_layout.addLayout(controls_row)
        main_layout.addWidget(self.frame_slider)

        # Connect UI signals to event handlers
        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button.clicked.connect(self.next_frame)
        self.jump_prev_button.clicked.connect(lambda: self.jump_frames(-60))
        self.jump_next_button.clicked.connect(lambda: self.jump_frames(60))
        self.frame_slider.sliderMoved.connect(self.slider_moved)
        self.show_trails.stateChanged.connect(self.update_image)
        self.show_boxes.stateChanged.connect(self.update_image)

    def update_tracking(self):

        if self.current_frame not in self.processed_frames:
            frame = self.frames[self.current_frame]
            try:
                tracked_objects = self.detector.update(
                    frame.copy()
                ) 
                print(f"Frame {self.current_frame}: {len(tracked_objects)} objects tracked")
                self.processed_frames[self.current_frame] = tracked_objects
            except Exception as e:
                print(f"Error processing frame {self.current_frame}: {e}")
                return
        self.update_image()

    def update_image(self):
    
        frame = self.frames[self.current_frame].copy()  
        h, w, c = frame.shape

        # Convert frame to QImage
        if c == 1:
            img = QtGui.QImage(frame, w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(frame, w, h, QtGui.QImage.Format_RGB888)

        pixmap = QtGui.QPixmap.fromImage(img)
        painter = QtGui.QPainter(pixmap)

        # Draw tracking results if available
        if self.current_frame in self.processed_frames:
            for obj in self.processed_frames[self.current_frame]:
                # Draw bounding box
                if self.show_boxes.isChecked() and "bbox" in obj:
                    minr, minc, maxr, maxc = obj["bbox"]
                    painter.setPen(QtGui.QPen(QtCore.Qt.green, 2))
                    painter.drawRect(minc, minr, maxc - minc, maxr - minr)

                # Draw movement trail
                if self.show_trails.isChecked() and len(obj["history"]) > 1:
                    painter.setPen(QtGui.QPen(QtCore.Qt.yellow, 2))
                    for i in range(1, len(obj["history"])):
                        prev = obj["history"][i - 1]
                        curr = obj["history"][i]
                        painter.drawLine(
                            int(prev[0]),
                            int(prev[1]),
                            int(curr[0]),
                            int(curr[1]),
                        )

                # Draw current position
                x, y = obj["position"]
                painter.drawEllipse(QtCore.QPoint(int(x), int(y)), 5, 5)

        painter.end()
        self.img_label.setPixmap(pixmap)
        self.frame_slider.setValue(self.current_frame)

    def prev_frame(self):

        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_tracking()

    def next_frame(self):

        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            self.update_tracking()

    def jump_frames(self, delta):
       
        new_frame = self.current_frame + delta
        new_frame = max(0, min(new_frame, len(self.frames) - 1))

        if new_frame != self.current_frame:
            self.current_frame = new_frame
            # Load necessary frame history
            if new_frame > 0:
                prev_frame = new_frame - 1
                if prev_frame not in self.processed_frames:
                    self.detector.prev_frame = self.frames[prev_frame] if self.frames[prev_frame].ndim == 2 else rgb2gray(self.frames[prev_frame])
            if new_frame > 1:
                prev_prev_frame = new_frame - 2
                if prev_prev_frame not in self.processed_frames:
                    self.detector.prev_prev_frame = self.frames[prev_prev_frame] if self.frames[prev_prev_frame].ndim == 2 else rgb2gray(self.frames[prev_prev_frame])
            self.update_tracking()

    def slider_moved(self, position):
    
        if position != self.current_frame:
            self.current_frame = position
            # Load necessary frame history
            if position > 0:
                prev_frame = position - 1
                if prev_frame not in self.processed_frames:
                    self.detector.prev_frame = self.frames[prev_frame] if self.frames[prev_frame].ndim == 2 else rgb2gray(self.frames[prev_frame])
            if position > 1:
                prev_prev_frame = position - 2
                if prev_prev_frame not in self.processed_frames:
                    self.detector.prev_prev_frame = self.frames[prev_prev_frame] if self.frames[prev_prev_frame].ndim == 2 else rgb2gray(self.frames[prev_prev_frame])
            self.update_tracking()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Object tracking demo with Kalman filters"
    )
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=-1,
        help="Number of frames to process (-1 for all)",
    )
    parser.add_argument(
        "--grey", action="store_true", help="Convert video to grayscale"
    )

    args = parser.parse_args()

    # Load video
    try:
        if args.num_frames > 0:
            frames = vread(args.video_path, num_frames=args.num_frames, as_grey=args.grey)
        else:
            frames = vread(args.video_path, as_grey=args.grey)
    except Exception as e:
        print(f"Error loading video: {e}")
        sys.exit(1)

    print(f"Loaded video with {len(frames)} frames")

    # Run application
    app = QtWidgets.QApplication([])
    window = QtDemo(frames)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())