# Rency Ajit Kansagra
import numpy as np
import skimage.io as io
from color_space_test import rgb_to_hsv, hsv_to_rgb
from numpy.lib import stride_tricks

def random_square_crop(image, size):
    h, w = image.shape[:2]
    
    if not (0 < size <= min(h, w)):
        raise ValueError(f"Size must be between 0 and {min(h, w)}")
    
    # Randomly select the top-left coordinate (x, y) for the crop
    x = np.random.randint(0, w - size + 1)
    y = np.random.randint(0, h - size + 1)

    return image[y:y+size, x:x+size]


def extract_patch(img, num_patches):
    
    size = num_patches
    img_size = img.shape

    
    if img_size[0] != img_size[1]:
        min_dimension = min(img_size[0], img_size[1])
        img = random_square_crop(img, min_dimension)  # Ensures square shape

    H, W = img.shape[:2]
    patch_size = H // size  # Compute patch size

 
    shape = (size, size, patch_size, patch_size, img.shape[2])

    # Define strides to efficiently extract patches
    strides = (patch_size * img.strides[0], patch_size * img.strides[1]) + img.strides

  
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)

    # Save patches as individual images
    for i in range(size):
        for j in range(size):
            filename = f"patch_{i}{j}.jpeg"
            io.imsave(filename, patches[i, j])

    print("Image patches added to the folder")

def resize_img(img, factor):
    
    if factor <= 0:
        raise ValueError("Scale factor must be greater than 0.")

    H, W = img.shape[:2]
    new_H, new_W = int(H * factor), int(W * factor)

    # Mapping new pixel positions to the nearest original pixel
    row_indices = np.floor(np.arange(new_H) / factor).astype(int)
    col_indices = np.floor(np.arange(new_W) / factor).astype(int)
    
    row_indices = np.clip(row_indices, 0, H - 1)
    col_indices = np.clip(col_indices, 0, W - 1)
  
    resized_img = img[row_indices[:, None], col_indices]

    output_filename = "resized_image.jpeg"
    io.imsave(output_filename, resized_img)
    print(f"Resized image saved as '{output_filename}'")

    return resized_img



def color_jitter(img, hue, saturation, value):
   
    # Convert RGB to HSV using custom function
    hsv_img = rgb_to_hsv(img.astype(np.float32))  # Ensure float32 for calculations

    # Generate random perturbation values
    h_perturb = np.random.uniform(-hue / 360, hue / 360, size=hsv_img.shape[:2])  # Normalize hue changes
    s_perturb = np.random.uniform(-saturation, saturation, size=hsv_img.shape[:2])
    v_perturb = np.random.uniform(-value, value, size=hsv_img.shape[:2])

 
    hsv_img[..., 0] = ((hsv_img[..., 0] + h_perturb) % 1)  # Keep hue in [0,1]
    hsv_img[..., 1] = np.clip(hsv_img[..., 1] + s_perturb, 0, 1)  # Saturation in [0,1]
    hsv_img[..., 2] = np.clip(hsv_img[..., 2] + v_perturb, 0, 1)  # Value in [0,1]

    jittered_img = hsv_to_rgb(hsv_img)

    jittered_img = (jittered_img * 255).astype(np.uint8)
    io.imsave("img_color_jitter.jpeg", jittered_img)
    print("New img_color_jitter.jpeg added to folder.")

    return jittered_img
   

# Uncomment the lines accordingly to perform different operations.

img = io.imread("img1.jpg")

#cropped_img = random_square_crop(img, 300)
#extract_patch(img, 4)
#resized_img = resize_img(img, 8)
jittered_img = color_jitter(img, 30, 0.1,0.2)
#io.imsave("cropped.jpg", cropped_img)
