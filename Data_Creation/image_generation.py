import os
import cv2
import numpy as np
import random

def rotate_images(folder_path, angle_range=(-7, 7), step=2, scale_factor=1):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # Count the number of original images
    num_original_images = sum(1 for filename in os.listdir(folder_path) if filename.endswith(".png"))

    # Start numbering for the rotated images
    image_counter = num_original_images + 1

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            for angle in range(angle_range[0], angle_range[1], step):
                # Rotate the image
                rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
                rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

                # Create a canvas with scaled size
                scaled_width = int(rotated_image.shape[1] * scale_factor)
                scaled_height = int(rotated_image.shape[0] * scale_factor)
                canvas = np.ones((scaled_height, scaled_width, 3), dtype=np.uint8) * 255

                # Calculate the offset to paste the rotated image at the center
                offset_x = (scaled_width - rotated_image.shape[1]) // 2
                offset_y = (scaled_height - rotated_image.shape[0]) // 2

                # Paste the rotated image onto the canvas
                canvas[offset_y:offset_y+rotated_image.shape[0], offset_x:offset_x+rotated_image.shape[1]] = rotated_image

                # Save the rotated image with white background and numbered filename
                rotated_filename = f"{image_counter}.png"
                rotated_image_path = os.path.join(folder_path, rotated_filename)
                cv2.imwrite(rotated_image_path, canvas)

                # Increment the image counter
                image_counter += 1

## add noise

def generate_variety_images(folder_path, scale_factors=[0.8, 0.85, 0.9, 0.95], jpeg_quality=80):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    num_existing_images = sum(1 for _ in os.listdir(folder_path) if _.endswith('.png'))
    image_counter = num_existing_images + 5

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Generate images with different qualities
            for scale_factor in scale_factors:
                resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)  
                image_name = f"{image_counter}.png"
                image_path = os.path.join(folder_path, image_name)
                cv2.imwrite(image_path, blurred_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

                # Increment the image counter
                image_counter += 1


def ensure_image_count(folder_path, target_image_count=2000):
    current_image_count = len(os.listdir(folder_path))
    if current_image_count > target_image_count:
        images_to_delete = current_image_count - target_image_count
        print(f"Folder {folder_path} has {current_image_count} images. Deleting {images_to_delete} extra images.")

        # Get all image filenames
        image_filenames = os.listdir(folder_path)
        
        # Select random images to delete
        images_to_delete_filenames = random.sample(image_filenames, images_to_delete)
        
        # Delete the selected images
        for filename in images_to_delete_filenames:
            os.remove(os.path.join(folder_path, filename))