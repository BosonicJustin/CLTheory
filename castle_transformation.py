import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load image using PIL for better color handling"""
    return np.array(Image.open(image_path))

def rotate_image(image, angle):
    """Rotate image while preserving color information"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to fit the rotated image
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Apply rotation with white background to preserve colors
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return rotated

def warp_image(image):
    """Apply perspective warping transformation"""
    height, width = image.shape[:2]
    
    # Define source points (corners of original image)
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    
    # Define destination points for warping effect
    offset = width * 0.15
    dst_points = np.float32([[offset, 0], [width - offset, offset], 
                            [0, height], [width, height - offset]])
    
    # Get perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply warping
    warped = cv2.warpPerspective(image, matrix, (width, height), 
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return warped

def resize_to_same_height(images, scale_factor=0.8):
    """Resize all images to have the same height and scale them down"""
    min_height = min(img.shape[0] for img in images)
    scaled_height = int(min_height * scale_factor)
    resized_images = []
    
    for img in images:
        height, width = img.shape[:2]
        new_width = int(width * scaled_height / height)
        resized = cv2.resize(img, (new_width, scaled_height))
        resized_images.append(resized)
    
    return resized_images

def stitch_images_horizontally(images, spacing=30):
    """Stitch images horizontally with spacing"""
    # Resize to same height and scale down
    images = resize_to_same_height(images)
    
    # Calculate total width needed
    total_width = sum(img.shape[1] for img in images) + spacing * (len(images) - 1)
    height = images[0].shape[0]
    
    # Create white canvas
    stitched = np.full((height, total_width, 3), 255, dtype=np.uint8)
    
    # Place images with spacing
    x_offset = 0
    for img in images:
        img_width = img.shape[1]
        stitched[:, x_offset:x_offset + img_width] = img
        x_offset += img_width + spacing
    
    return stitched

def add_colored_borders(images, colors):
    """Add colored borders around images"""
    bordered_images = []
    
    for img, color in zip(images, colors):
        height, width = img.shape[:2]
        
        # Border thickness
        border_thickness = 15
        
        # Create a larger canvas with the border color
        new_height = height + 2 * border_thickness
        new_width = width + 2 * border_thickness
        
        # Create border canvas with the specified color
        bordered_img = np.full((new_height, new_width, 3), color, dtype=np.uint8)
        
        # Place the original image in the center
        bordered_img[border_thickness:border_thickness + height, 
                    border_thickness:border_thickness + width] = img
        
        bordered_images.append(bordered_img)
    
    return bordered_images

def main():
    # Load the castle image
    image_path = "test_image/Gedimino_pilis_by_Augustas_Didzgalvis.jpg"
    original_image = load_image(image_path)
    
    # Create 4 versions with different transformations
    images = []
    colors = [(0, 100, 255), (0, 200, 0), (255, 50, 50), (255, 50, 50)]  # blue, green, red, red
    
    # 1. Original image
    images.append(original_image)
    
    # 2. Slight rotation (20 degrees)
    slightly_rotated = rotate_image(original_image, 20)
    images.append(slightly_rotated)
    
    # 3. Warped image
    warped = warp_image(original_image)
    images.append(warped)
    
    # 4. Very steep rotation (75 degrees)
    steep_rotated = rotate_image(original_image, 75)
    images.append(steep_rotated)
    
    # Add colored borders to images
    labeled_images = add_colored_borders(images, colors)
    
    # Stitch all images together
    final_image = stitch_images_horizontally(labeled_images)
    
    # Save the result
    output_image = Image.fromarray(final_image)
    output_image.save("castle_transformations.jpg", quality=95)
    
    # Display the result
    plt.figure(figsize=(20, 10))
    plt.imshow(final_image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("castle_transformations_display.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Transformation complete!")
    print(f"Final image shape: {final_image.shape}")
    print("Saved as: castle_transformations.jpg")

if __name__ == "__main__":
    main()