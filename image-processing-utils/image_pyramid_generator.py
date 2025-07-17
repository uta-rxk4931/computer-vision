# Rency Ajit Kansagra
import numpy as np
import cv2
import os
from PIL import Image

def image_pyramid(image_array, pyramid_height,output_filename):
    
    # Convert numpy array to PIL image
    image = Image.fromarray(image_array)

    # Get image size
    width, height = image.size
    base_name, ext = os.path.splitext(output_filename)
    # Generate and save pyramid images
    for i in range(1, pyramid_height):
        factor = 2 ** i
        size = (width // factor, height // factor)

        if size[0] < 1 or size[1] < 1:
            break  # Stop when image gets too small

        resized = image.resize(size, Image.Resampling.LANCZOS) # Using a resampling filter to resize images
        # Create output filename with scale factor
        output_path = f"{base_name}_{factor}x{ext}"
        
        # Save resized image
        resized.save(output_path)
        print(f"Saved {output_path} ({size[0]}x{size[1]})")

# Example usage
if __name__ == "__main__":

    # Load an image as a numpy array
    img_path = "img.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

    # Call function with numpy image array and pyramid height
    image_pyramid(img, 4,img_path)
