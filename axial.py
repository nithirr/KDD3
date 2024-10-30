import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_kidney(image_path):
    # Load the original image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the grayscale image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply a binary threshold to the blurred image
    _, binary_mask = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the kidney
    kidney_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the kidney
    kidney_mask = np.zeros_like(gray_image)
    cv2.drawContours(kidney_mask, [kidney_contour], -1, 255, thickness=cv2.FILLED)

    # Segment the kidney from the original image using the mask
    kidney_segmented = cv2.bitwise_and(image, image, mask=kidney_mask)

    # Display the results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title('Binary Mask')
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('Refined Mask')
    plt.imshow(refined_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title('Segmented Kidney')
    plt.imshow(cv2.cvtColor(kidney_segmented, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

    return kidney_segmented

# Replace 'kidney_image.jpg' with the path to your kidney image
kidney_segmented = segment_kidney('sam_img/Normal- (28).jpg')
