import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.util import random_noise
from skimage.restoration import denoise_nl_means, estimate_sigma
from PIL import Image
import cv2
import os

def load_image(image_path):
    """Load an image from a file path and convert to float."""
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return img_as_float(image)

def save_image(image, output_path):
    """Save a float image to a file path."""
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(output_path)

def denoise_image(image_path, output_path):
    """Load an image, add Poisson noise, denoise it, and save the result."""
    # Load image
    image = load_image(image_path)

    # Add Poisson noise to the image
    noisy_image = random_noise(image, mode='poisson')

    # Estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(noisy_image, channel_axis=None))

    # Apply Non-Local Means Denoising
    patch_kw = dict(patch_size=5,      # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    channel_axis=None)

    denoised_image = denoise_nl_means(noisy_image, h=1.15 * sigma_est, fast_mode=True, **patch_kw)
    cv2.imshow("img",denoised_image)
    #cv2.imwrite("sam_img/noise/noise_rem_4.png",denoised_image)

    # Save the denoised image
    save_image(denoised_image, output_path)

    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')

    ax[1].imshow(noisy_image, cmap='gray')
    ax[1].set_title('Image with Poisson Noise')

    ax[2].imshow(denoised_image, cmap='gray')
    ax[2].set_title('Denoised Image')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage:
input_image_path = 'sam_img/raw/Normal- (28).jpg'  # Replace with your image path
output_image_path = 'sam_img/noise/noise_rem_1.png'  # Replace with your output path

denoise_image(input_image_path, output_image_path)
