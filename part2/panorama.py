import numpy as np
import cv2
import imutils
import os
import matplotlib.pyplot as plt

def load_images_from_folder(folder, output_path):
    """Loads all images from a given folder and returns them as a list."""
    files = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    images = []
    for file in files:
        path_to_image = os.path.join(folder, file)
        img = cv2.imread(path_to_image)
        if img is not None:
            images.append(img)
            cv2.imwrite(os.path.join(output_path, f"loaded_{file}"), img)
    return images

def detect_and_draw_keypoints(images, output_path):
    """Detects ORB keypoints in images and saves them."""
    orb = cv2.ORB_create()
    for i, img in enumerate(images):
        keypoints = orb.detect(img, None)
        keypoints_image = cv2.drawKeypoints(img, keypoints, None, color=(0, 0, 255))
        cv2.imwrite(os.path.join(output_path, f"keypoints_{i+1}.png"), keypoints_image)

def stitch_images(images, output_path):
    """Stitches images together and returns the stitched image."""
    image_stitcher = cv2.Stitcher_create()
    error, stitched_image = image_stitcher.stitch(images)
    if error:
        print("Error: Could not stitch images")
        return None
    cv2.imwrite(os.path.join(output_path, "stitched_raw.png"), stitched_image)
    return stitched_image

def crop_stitched_image(stitched_image, output_path):
    """Crops the stitched image to remove unnecessary black areas."""
    stitched_image = cv2.copyMakeBorder(stitched_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    thresh_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    area_oi = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh_image.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(area_oi)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    min_rectangle = mask.copy()
    sub = mask.copy()
    while cv2.countNonZero(sub) > 0:
        min_rectangle = cv2.erode(min_rectangle, None)
        sub = cv2.subtract(min_rectangle, thresh_image)
    contours = cv2.findContours(min_rectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    area_oi = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(area_oi)
    cropped_image = stitched_image[y:y + h, x:x + w]
    cv2.imwrite(os.path.join(output_path, "stitched_cropped.png"), cropped_image)
    return cropped_image

def main():
    input_path = input("Enter the input directory: ")
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)
    for file in os.listdir(output_path):
        os.remove(os.path.join(output_path, file))
    
    images = load_images_from_folder(input_path, output_path)
    if not images:
        print("No images found.")
        return
    
    detect_and_draw_keypoints(images, output_path)
    stitched_image = stitch_images(images, output_path)
    
    if stitched_image is not None:
        cropped_image = crop_stitched_image(stitched_image, output_path)
        print("Processing complete. Check the 'outputs' directory for results.")

if __name__ == "__main__":
    main()
