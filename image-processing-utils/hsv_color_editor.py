# Rency Ajit Kansagra
import numpy as np
import cv2
import argparse
import sys

import numpy as np

def rgb_to_hsv(rgb_img):
    # Normalize RGB values to [0, 1]
    rgb = rgb_img.astype(np.float32) / 255.0
    R, G, B = np.moveaxis(rgb, -1, 0)

    V = np.maximum.reduce([R, G, B])
    C = V - np.minimum.reduce([R, G, B])
    
    S = np.divide(C, V, where=V != 0, out=np.zeros_like(V))

    H = np.full_like(V, 0)
    nonzero_C = C != 0

    cond_r = (V == R) & nonzero_C # If max is R
    cond_g = (V == G) & nonzero_C # If max is G
    cond_b = (V == B) & nonzero_C # If max is B

    H[cond_r] = (60 * ((G[cond_r] - B[cond_r]) / C[cond_r]) % 360) # Case where V == R
    H[cond_g] = (60 * ((B[cond_g] - R[cond_g]) / C[cond_g]) + 120) # Case where V == G
    H[cond_b] = (60 * ((R[cond_b] - G[cond_b]) / C[cond_b]) + 240) # Case where V == B

    return np.stack([H, S, V], axis=-1)


def hsv_to_rgb(hsv_img):
    H, S, V = hsv_img[..., 0], hsv_img[..., 1], hsv_img[..., 2]
    C = V * S
    X = C * (1 - np.abs((H / 60) % 2 - 1))
    m = V - C
    # Determine the sector of Hue and assign RGB values accordingly
    idx = ((H // 60) % 6).astype(int)
    
    r = np.choose(idx, [C, X, 0, 0, X, C])
    g = np.choose(idx, [X, C, C, X, 0, 0])
    b = np.choose(idx, [0, 0, X, C, C, X])
    #Add m to shift RGB values and scale back to [0, 255]
    rgb = np.stack([r, g, b], axis=-1) + m[..., np.newaxis]
    
    return (rgb * 255).clip(0, 255).astype(np.uint8)

def modify_image(filename, h_mod, s_mod, v_mod):

    # Read image (BGR format)
    img = cv2.imread(filename)
    if img is None:
        print(f"Error: Could not read image file '{filename}'")
        sys.exit(1)
    
  
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = rgb_to_hsv(rgb_img)
    
    # Modify HSV values
    hsv_img[..., 0] = (hsv_img[..., 0] + h_mod) % 360  # Hue
    hsv_img[..., 1] = np.clip(hsv_img[..., 1] * s_mod, 0, 1)  # Saturation
    hsv_img[..., 2] = np.clip(hsv_img[..., 2] * v_mod, 0, 1)  # Value
    
  
    modified_rgb = hsv_to_rgb(hsv_img)
    modified_bgr = cv2.cvtColor(modified_rgb, cv2.COLOR_RGB2BGR)
    output_filename = f"modified_{filename}"
    
    # Save the modified image
    cv2.imwrite(output_filename, modified_bgr)
    print(f"Modified image saved as '{output_filename}'")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Modify image HSV values')
    parser.add_argument('filename', help='Input image file')
    parser.add_argument('hue', type=float, help='Hue modification (degrees)')
    parser.add_argument('saturation', type=float, help='Saturation modification')
    parser.add_argument('value', type=float, help='Value modification')
    
    args = parser.parse_args()
    
    if not (0 <= args.hue <= 360):
        print("Error: Hue must be in range [0, 360]")
        sys.exit(1)
    
    if not (0 <= args.saturation <= 1):
        print("Error: Saturation must be in range [0, 1]")
        sys.exit(1)
    
    if not (0 <= args.value <= 1):
        print("Error: Value must be in range [0, 1]")
        sys.exit(1)
    
    # Process the image
    modify_image(args.filename, args.hue, args.saturation, args.value)

if __name__ == "__main__":
    main()

