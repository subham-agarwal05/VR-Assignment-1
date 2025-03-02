import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"No image found at: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(kp1, des1, kp2, des2, threshold=0.75):
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in raw_matches if m.distance < threshold * n.distance]
    if len(good_matches) < 4:
        return None
    return np.array([[kp1[m.queryIdx].pt + kp2[m.trainIdx].pt] for m in good_matches]).reshape(-1, 4)

def stitch_images(base_img, target_img, matches):
    if matches is None or len(matches) < 4:
        raise ValueError("Not enough matches to compute homography")
    
    points1, points2 = matches[:, :2], matches[:, 2:]
    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography computation failed")
    
    height, width = base_img.shape[:2]
    result = cv2.warpPerspective(target_img, H, (width + target_img.shape[1], height))
    result[0:height, 0:width] = base_img
    return result

def crop_tilted_black_seam(image):
    """ Detects and removes the tilted black seam from the stitched image. """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold to detect black pixels
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the black region
    contours, _ = cv2.findContours(255 - thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image  # No black regions found, return original image

    # Find the largest black region (the seam)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding rotated rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Get the bounding box and crop along its edges
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = image[:, :x]  # Crop everything before the seam starts

    return cropped



def stitch_multiple_images(image_paths):
    images = [load_image(path) for path in image_paths]
    
    # Resize images to half for better processing speed
    images = [cv2.resize(image, (0, 0), fx=0.5, fy=0.5) for image in images]
    
    stitched_image = images[0]

    for i in range(1, len(images)):
        kp1, des1 = extract_sift_features(cv2.cvtColor(stitched_image, cv2.COLOR_RGB2GRAY))
        kp2, des2 = extract_sift_features(cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY))
        
        matches = match_keypoints(kp1, des1, kp2, des2, threshold=0.75)
        if matches is None:
            matches = match_keypoints(kp1, des1, kp2, des2, threshold=0.9)

        if matches is None:
            print(f"Skipping image {i} due to insufficient matches.")
            continue

        try:
            # Stitch images
            stitched_image = stitch_images(stitched_image, images[i], matches)
            
            # Remove the tilted black seam
            # stitched_image = crop_tilted_black_seam(stitched_image)

        except ValueError:
            print(f"Homography failed for image {i}. Skipping...")
            continue

    return stitched_image


image_folder = input("Enter the input directory: ")
output_folder = 'output/'
os.makedirs(output_folder, exist_ok=True)
for file in os.listdir(output_folder):
    os.remove(os.path.join(output_folder, file))
image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
if len(image_paths) < 2:
    raise ValueError("At least two images are required for stitching.")
stitched_result = stitch_multiple_images(image_paths)
cv2.imwrite(os.path.join(output_folder, 'stitched_result.jpeg'), cv2.cvtColor(stitched_result, cv2.COLOR_RGB2BGR))
plt.figure(figsize=(10, 5))
plt.imshow(stitched_result)
plt.axis('off')
plt.show()
