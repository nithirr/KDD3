import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_clahe(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded properly
    if img is None:
        print("Error: Image not loaded properly.")
        return

    # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to the image
    clahe_img = clahe.apply(img)
    cv2.imshow("clahe",clahe_img)
    cv2.imwrite("sam_img/clahe_4.png",clahe_img)
    # Plotting the images
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # CLAHE enhanced image
    plt.subplot(1, 2, 2)
    plt.title('CLAHE Enhanced Image')
    plt.imshow(clahe_img, cmap='gray')
    plt.axis('off')

    plt.show()

# Example usage

image_path = 'sam_img/Tumor- (1203).jpg'  # Replace with your image path
apply_clahe(image_path)
