import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

def load_and_normalize_images(image_dir):
    """Load images and normalize them to the same height."""
    # Get all image files
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")) + 
                        glob(os.path.join(image_dir, "*.jpeg")))
    
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        images.append(img)
    
    if not images:
        raise ValueError("No images found in the directory")
    
    # Find the minimum height to normalize all images, then scale up
    min_height = min(img.shape[0] for img in images)
    target_height = int(min_height * 1.5)  # Make images 50% taller
    
    # Resize all images to have the same height while maintaining aspect ratio
    normalized_images = []
    for img in images:
        height, width = img.shape[:2]
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        
        resized_img = cv2.resize(img, (new_width, target_height))
        normalized_images.append(resized_img)
    
    return normalized_images

def merge_images_horizontally(images, border_width=3):
    """Merge images horizontally with borders between them."""
    if len(images) == 1:
        return images[0]
    
    # Create border (black line)
    border_color = [0, 0, 0]  # Black border
    height = images[0].shape[0]
    
    merged_parts = [images[0]]
    
    for img in images[1:]:
        # Add white border
        border = np.full((height, border_width, 3), border_color, dtype=np.uint8)
        merged_parts.append(border)
        merged_parts.append(img)
    
    return np.hstack(merged_parts)

def display_merged_images(image_dir):
    """Load, normalize, merge and display images side-by-side."""
    try:
        # Load and normalize images
        images = load_and_normalize_images(image_dir)
        print(f"Loaded {len(images)} images")
        
        # Print image dimensions for verification
        for i, img in enumerate(images):
            print(f"Image {i+1}: {img.shape[1]}x{img.shape[0]} (WxH)")
        
        # Merge images horizontally with borders
        merged_image = merge_images_horizontally(images)
        print(f"Merged image dimensions: {merged_image.shape[1]}x{merged_image.shape[0]} (WxH)")
        
        # Display the merged image
        plt.figure(figsize=(15, 12))
        plt.imshow(merged_image)
        plt.axis('off')  # Remove axes for clean display
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    image_directory = "images_castle/"
    display_merged_images(image_directory)