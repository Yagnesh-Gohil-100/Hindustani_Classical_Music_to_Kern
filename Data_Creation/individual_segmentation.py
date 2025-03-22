import cv2
import numpy as np
from PIL import Image
import os

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding on grayscale image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

    return thresh

def separate_articulation(image):
    """
    Function to separate articulation in an image.
    
    Parameters:
    - image: numpy array representing the input image.
    
    Returns:
    - upper_part: The upper part of the image if articulation is separated.
    - original_image: The original image if no separation was done.
    """

    # Process the image
    processed_image = preprocess_image(image)
    
    # Find contours
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < h < 21 and w > 25:
            # Segmentation logic based on detected contour
            # alphabet_region = image[y:y+h, x:x+w]
            
            # Crop the original image along the y-coordinate
            upper_part = image[:y, :]
            # lower_part = image[y:, :]
            
            # Check if separation is successful
            if upper_part.shape[0] > 0:
                return upper_part
            
            break  # Assuming only one contour meets the criteria per image
    
    # If no separation was done, return the original image
    return image


# meend and kann swar segmentation

def extract_alphabets_vertical(image_path):
    """
    Function to perform vertical segmentation on an image.
    
    Parameters:
    - image: numpy array representing the input image.
    
    Returns:
    - left_part: The left part of the image if separation was successful, otherwise None.
    - mid_part: The mid part of the image. If no separation, returns the original image.
    - right_part: The right part of the image if separation was successful, otherwise None.
    """

    image = cv2.imread(image_path)
    if image is None:
        return None, None, None

    processed_image = preprocess_image(image)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_coords = []
    all_coords = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 <= h <= 25 and w > 25:
            valid_coords.append((x, y, w, h))
        else:
            all_coords.append((x, y, w, h))

    left_part, mid_part, right_part = None, None, None
    
    if valid_coords:
        valid_coords = sorted(valid_coords, key=lambda coord: coord[0])
        leftmost_valid = valid_coords[0]
        rightmost_valid = valid_coords[-1]

        left_cut = None
        x1, y1, w1, h1 = leftmost_valid
        for x, y, w, h in all_coords:
            if x <= x1 and (y + h) >= y1 and h > 10 and w > 10:
                left_cut = x1
                break

        right_cut = None
        x2, y2, w2, h2 = rightmost_valid
        for x, y, w, h in all_coords:
            if x >= (x2 + w2) and (y + h) >= y2 and h > 10 and w > 10:
                right_cut = x2 + w2
                break

        if left_cut is not None and right_cut is not None:
            left_part = image[:, :left_cut]
            mid_part = image[:, left_cut:right_cut]
            right_part = image[:, right_cut:]

        elif left_cut is not None:
            left_part = image[:, :left_cut]
            right_part = image[:, left_cut:]
        
        elif right_cut is not None:
            left_part = image[:, :right_cut]
            right_part = image[:, right_cut:]

    # If no segmentation occurred, treat the entire image as mid_part
    if left_part is None and right_part is None:
        mid_part = image

    return left_part, mid_part, right_part

def identify_meend_and_kann_swar(left_part, mid_part, right_part):
    """
    Function to identify and structure meend and kann swar based on the width of the segments.
    
    Parameters:
    - left_part: The left part of the image.
    - mid_part: The mid part of the image.
    - right_part: The right part of the image.
    
    Returns:
    - left_part: The left part (kann swar or None).
    - mid_part: The mid part (meend).
    - right_part: The right part (kann swar or None).
    """
    # Determine which part is meend based on width
    parts = {
        "left": left_part,
        "mid": mid_part,
        "right": right_part
    }

    # Filter out None parts
    valid_parts = {k: v for k, v in parts.items() if v is not None}

    # If no segmentation occurred (only mid_part exists)
    if len(valid_parts) == 1 and "mid" in valid_parts:
        # Treat the entire image as meend
        left_part = None
        right_part = None
        mid_part = valid_parts["mid"]
    
    # If there are only two parts, identify meend based on width
    elif len(valid_parts) == 2:
        # Find the part with the maximum width (meend)
        meend_key = max(valid_parts, key=lambda k: valid_parts[k].shape[1])
        kann_swar_key = [k for k in valid_parts.keys() if k != meend_key][0]

        # Reassign parts to ensure meend is in the middle
        if meend_key == "left":
            mid_part = valid_parts[meend_key]
            right_part = valid_parts[kann_swar_key]
            left_part = None
        elif meend_key == "right":
            mid_part = valid_parts[meend_key]
            left_part = valid_parts[kann_swar_key]
            right_part = None
        else:
            # If meend is already in the middle, no changes needed
            pass

    # If there are three parts, meend is always in the middle
    elif len(valid_parts) == 3:
        mid_part = valid_parts["mid"]
        left_part = valid_parts["left"]
        right_part = valid_parts["right"]

    return left_part, mid_part, right_part

def save_segments(left_part, mid_part, right_part, output_folder, image_file):
    """
    Function to save the segmented parts.
    
    Parameters:
    - left_part: The left part of the image.
    - mid_part: The mid part of the image.
    - right_part: The right part of the image.
    - output_folder: The folder to save the segmented images.
    - image_file: The name of the original image file.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save mid_part (meend) regardless of segmentation
    if mid_part is not None:
        mid_part_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_mid.png")
        cv2.imwrite(mid_part_path, mid_part)
    
    # Save left_part (kann swar) if it exists
    if left_part is not None:
        left_part_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_left.png")
        cv2.imwrite(left_part_path, left_part)
    
    # Save right_part (kann swar) if it exists
    if right_part is not None:
        right_part_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_right.png")
        cv2.imwrite(right_part_path, right_part)

import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, square
from skimage import img_as_ubyte

# Function to display images 
# def display_image(img, cmap='gray'):
#     plt.figure(figsize=(10, 2))
#     plt.imshow(img, cmap=cmap)
#     plt.axis('off')
#     plt.show()

# Function to segment a single image and return the segmented parts
def segment_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply simple binary thresholding and invert the image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Define structuring elements
    structuring_element2 = np.ones((2, 2), dtype=bool)
    structuring_element_erosion = square(3)

    # Apply binary dilation to fill gaps
    dilated = binary_dilation(binary, footprint=structuring_element2)

    # Apply binary erosion to separate connected components
    eroded = binary_erosion(dilated, footprint=structuring_element_erosion)
    eroded = img_as_ubyte(eroded)  # Convert to uint8 for display purposes

    # Perform vertical projection to find potential cut lines
    vertical_projection = np.sum(eroded, axis=0)

    # Find cut points by identifying valleys in the projection with heuristic
    threshold = 0.15 * np.max(vertical_projection)
    valleys = [x for x, y in enumerate(vertical_projection) if y < threshold]

    # Apply heuristic: if two consecutive valleys are close, take the right one
    cut_points = []
    min_distance = 13
    i = 0
    while i < len(valleys) - 1:
        if (valleys[i + 1] - valleys[i]) < min_distance:
            cut_points.append(valleys[i + 1])
            i += 2  # Skip the next valley since we took the right one
        else:
            cut_points.append(valleys[i])
            i += 1
    if i == len(valleys) - 1:
        cut_points.append(valleys[i])  # Add the last valley if it's not processed

    # Ensure no duplicate cut points and sort them
    cut_points = sorted(set(cut_points))

    # Separate the image at cut points
    cut_images = []
    start = 0
    for cut_point in cut_points:
        if cut_point - start > 10:  # Ensure segments are large enough
            cut_image = img[:, start:cut_point]
            cut_images.append(cut_image)
            start = cut_point

    # Add the last segment
    cut_images.append(img[:, start:])

    return cut_images

def merge_segments(segments):
    final_images = []
    i = 0
    while i < len(segments):
        current_image = segments[i]
        current_ratio = current_image.shape[0] / current_image.shape[1]

        ratio_threshold = 1.8

        if current_image.shape[0] > 35:
            ratio_threshold = 2.9
        
        # If the ratio is greater than the threshold and it's the first segment
        if current_ratio > ratio_threshold and i == 0:
            # Merge with the next segment
            if i + 1 < len(segments):
                current_image = np.hstack((current_image, segments[i + 1]))
                final_images.append(current_image)
                i += 2
            else:
                final_images.append(current_image)
                i += 1
        # If two or more consecutive segments have a ratio greater than the threshold
        elif i < len(segments) - 1 and (segments[i+1].shape[0] / segments[i+1].shape[1]) > ratio_threshold:
            while i < len(segments) - 1 and (segments[i + 1].shape[0] / segments[i + 1].shape[1]) > ratio_threshold:
                current_image = np.hstack((current_image, segments[i + 1]))
                i += 1
            final_images.append(current_image)
            i += 1
        # If the ratio is greater than the threshold and it's not the first segment
        elif current_ratio > ratio_threshold and i != 0:
            # Merge with the previous segment
            if final_images:
                final_images[-1] = np.hstack((final_images[-1], current_image))
            else:
                final_images.append(current_image)
            i += 1
        else:
            final_images.append(current_image)
            i += 1

    return final_images

# Function to process a single image, segment, and save the results in the provided folder
def segment_word(image_file, original_folder):
    # Load the image
    img = cv2.imread(os.path.join(original_folder, image_file))
    
    # Segment the image
    segmented_images = segment_image(img)
    
    # Merge segments based on height-to-width ratio
    final_images = merge_segments(segmented_images)
    
    # Save the segmented images
    image_base_name = os.path.splitext(image_file)[0]
    for i, segmented_image in enumerate(final_images):
        cv2.imwrite(os.path.join(original_folder, f'{image_base_name}_{i+1}.png'), segmented_image)

# Example usage:
if __name__ == "__main__":
    # Change working directory to the script's location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # remove comments below for each segment one by one and run accordingly

    #-----------------------------------------------------------------------------------------------------------#

    # input_folder = "Example_Composites_Segmentation/jan_2025/articulation"  # Folder containing input images
    # output_folder = "Example_Composites_Segmentation/jan_2025/articulation_segmented"  # Folder to save processed images
    
    # # Create output folder if it doesn't exist
    # os.makedirs(output_folder, exist_ok=True)
    
    # # Iterate through all files in the input folder
    # for filename in os.listdir(input_folder):
    #     input_path = os.path.join(input_folder, filename)
        
    #     # Check if the file is an image
    #     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
    #         image = cv2.imread(input_path)
            
    #         # Process the image if it was read successfully
    #         if image is not None:
    #             result_image = separate_articulation(image)
                
    #             # Save the resulting image
    #             output_path = os.path.join(output_folder, filename)
    #             cv2.imwrite(output_path, result_image)
    #         else:
    #             print(f"Failed to read image: {input_path}")
    #     else:
    #         print(f"Skipping non-image file: {filename}")

    #-----------------------------------------------------------------------------------------------------------#

    # Example usage for separating meend from kann swar
    image_folder = "Example_Composites_Segmentation/compo_meend"  # Folder containing images
    output_folder = "Example_Composites_Segmentation/checking_compo_meend_2"  # Output folder

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        left_part, mid_part, right_part = extract_alphabets_vertical(image_path)
        
        # Identify and structure meend and kann swar
        left_part, mid_part, right_part = identify_meend_and_kann_swar(left_part, mid_part, right_part)
        
        # Save the segments
        save_segments(left_part, mid_part, right_part, output_folder, image_file)

    #-----------------------------------------------------------------------------------------------------------#

    # # Example of calling the process function for a single image and segmenting words (composite alphabets)
    # original_folder = "Example_Composites_Segmentation/jan_2025/words_demo"  # Replace with your input folder path

    # for image_file in os.listdir(original_folder):
    #     # Process the image and save the segmented parts
    #     segment_word(image_file, original_folder)
