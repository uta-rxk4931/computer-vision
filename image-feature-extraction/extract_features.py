import numpy as np
import cv2
from skimage.feature import hog, SIFT
from sklearn.cluster import KMeans
from tqdm import tqdm

def extract_sift(images, labels):
    """
    Extract SIFT descriptors from a list of images.

    """

    sift = SIFT()
    descriptors_list = []
    y_features = []
    
    for idx in tqdm(range(images.shape[0]), desc="Processing images"):
        try:
            sift.detect_and_extract(images[idx])
            if sift.descriptors is not None:
                descriptors_list.append(sift.descriptors)
                y_features.append(labels[idx])
        except:
            pass
    
    # Add SIFT statistics
    total_keypoints = sum(desc.shape[0] for desc in descriptors_list if desc is not None)
    print(f"\nSIFT Feature Statistics:")
    print(f"Total keypoints detected: {total_keypoints}")
    print(f"Average keypoints per image: {total_keypoints/len(descriptors_list):.2f}")
    print(f"SIFT descriptor size: {descriptors_list[0].shape[1]}")
    print(f"SIFT features extracted successfully from {len(descriptors_list)} images")
    
    return descriptors_list, y_features

def extract_hog(images):
    """
    Extract HOG features from a list of images.

    """

    hog_features = []
    
    for img in tqdm(images, desc="Extracting HOG features"):
        img_gray = cv2.cvtColor(img.reshape(32, 32, 3), cv2.COLOR_RGB2GRAY)
        features = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(features)
    
    hog_features = np.array(hog_features)
    print(f"\nHOG Feature Statistics:")
    print(f"Number of images processed: {len(hog_features)}")
    print(f"Features per image: {hog_features.shape[1]}")
    print(f"Total features extracted: {hog_features.size}")
    
    return hog_features

def create_bovw(descriptors_list, num_clusters=500):
    """
    Creates a bag of visual words model using k-means clustering.
    """
    print("\nCreating Bag of Visual Words...")
    
    all_descriptors = [desc for desc in descriptors_list if desc is not None]
    descriptors = np.vstack(all_descriptors)
    
    # Training KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(descriptors)
    print("KMeans training completed")

    # Creating feature histograms
    bovw_features = []
    for descriptors in tqdm(descriptors_list, desc="Creating BoVW features"):
        if descriptors is not None and len(descriptors) > 0:
            clusters = kmeans.predict(descriptors)
            histogram, _ = np.histogram(clusters, bins=num_clusters, range=(0, num_clusters))
            histogram = histogram.astype(np.float32)
        else:
            histogram = np.zeros(num_clusters, dtype=np.float32)
        bovw_features.append(histogram)
    
    bovw_features = np.array(bovw_features)
    
    # Applying TF-IDF transformation
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    bovw_features = tfidf.fit_transform(bovw_features).toarray()
    
    print("\nBoVW Feature Statistics:")
    print(f"Number of images encoded: {bovw_features.shape[0]}")
    print(f"Final feature vector size: {bovw_features.shape[1]}")
    print("BoVW creation completed")
    
    return kmeans, bovw_features


if __name__ == "__main__":
    # Load the pre-split data
    print("\nLoading CIFAR-10 dataset...")
    data = np.load("cifar10.npz", allow_pickle=True)
    X_train = data["X_train"].astype(np.uint8)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.uint8)
    y_test = data["y_test"]
    print(f"Dataset loaded - Train: {len(X_train)} images, Test: {len(X_test)} images")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training images shape: {X_train.shape}")
    print(f"Test images shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Convert images to grayscale
    X_train_gray = np.array([cv2.cvtColor(img.reshape(32, 32, 3), cv2.COLOR_RGB2GRAY) for img in X_train])
    X_test_gray = np.array([cv2.cvtColor(img.reshape(32, 32, 3), cv2.COLOR_RGB2GRAY) for img in X_test])
    
    # Extract SIFT features
    sift_descriptors_train, y_train_filtered = extract_sift(X_train_gray, y_train)
    sift_descriptors_test, y_test_filtered = extract_sift(X_test_gray, y_test)
    
    # Create bag of visual words
    kmeans_sift, X_train_sift = create_bovw(sift_descriptors_train)
    _, X_test_sift = create_bovw(sift_descriptors_test, num_clusters=kmeans_sift.n_clusters)
    
    # Extract HOG features
    X_train_hog = extract_hog(X_train)
    X_test_hog = extract_hog(X_test)
    
    # Save extracted features
    np.savez("sift_features.npz", X_train=X_train_sift, y_train=y_train_filtered, 
             X_test=X_test_sift, y_test=y_test_filtered)
    np.savez("hog_features.npz", X_train=X_train_hog, y_train=y_train, 
             X_test=X_test_hog, y_test=y_test)
    print("Features saved successfully!!!")
    
    # Print final feature dimensions
    print("\nFinal Feature Dimensions:")
    print(f"SIFT-BoVW features shape - Train: {X_train_sift.shape}, Test: {X_test_sift.shape}")
    print(f"HOG features shape - Train: {X_train_hog.shape}, Test: {X_test_hog.shape}")
    
    print("\nFeature extraction completed!")