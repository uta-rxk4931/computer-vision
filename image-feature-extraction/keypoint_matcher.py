import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import SIFT, match_descriptors
from skimage.color import rgb2gray, rgba2rgb
from skimage.io import imread

def extract_sift_features(image):
    """
    Extracts SIFT keypoints and descriptors from an image.

    """
    sift = SIFT()
    sift.detect_and_extract(image)
    return sift.keypoints, sift.descriptors

def match_keypoints(descriptors1, descriptors2):
    """
    Matches keypoints between two sets of SIFT descriptors.

    """
    matches = []
    for i, desc1 in enumerate(descriptors1):
        distances = np.linalg.norm(descriptors2 - desc1, axis=1)
        j = np.argmin(distances)
        matches.append((i, j))
    return matches

def match_keypoints_test(descriptors1, descriptors2):
    """
    Matches keypoints between two sets of SIFT descriptors.

    """
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
    return matches

def plot_keypoint_matches(img1, keypoints1, img2, keypoints2, matches):
    """
    Plots keypoint matches between two images.
    
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    img_combined = np.hstack((img1, img2))
    ax.imshow(img_combined, cmap='gray')
    ax.axis('off')
    
    cols1 = img1.shape[1]
    
    for i, j in matches:
        pt1 = keypoints1[i]
        pt2 = keypoints2[j]
        pt2_shifted = (pt2[0], pt2[1] + cols1)
        ax.plot([pt1[1], pt2_shifted[1]], [pt1[0], pt2_shifted[0]], 'r-', linewidth=0.8)
        ax.scatter([pt1[1], pt2_shifted[1]], [pt1[0], pt2_shifted[0]], c='yellow', s=10)
    
    plt.show()


img1 = imread("Rainier1.png")
if img1.shape[-1] == 4:
    img1 = rgba2rgb(img1)

img2 = imread("Rainier2.png")
if img2.shape[-1] == 4:
    img2 = rgba2rgb(img2)

# Convert to grayscale
img1 = rgb2gray(img1)
img2 = rgb2gray(img2)

# Extract keypoints and descriptors
keypoints1, descriptors1 = extract_sift_features(img1)
keypoints2, descriptors2 = extract_sift_features(img2)


# Match keypoints
matches = match_keypoints(descriptors1, descriptors2)

# Comment the above statement and uncomment the below statement to test with match_descriptors from skimage.feature.
# matches = match_keypoints_test(descriptors1, descriptors2)

# Print keypoints count
print(f"Detected {len(keypoints1)} keypoints in Image 1")
print(f"Detected {len(keypoints2)} keypoints in Image 2")
# Plot matches
plot_keypoint_matches(img1, keypoints1, img2, keypoints2, matches)