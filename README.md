# AIM 825- Sec-A: Visual Recognition Assignment 1

# Part 1



## Overview
This project detects, segments, and counts coins in an image using OpenCV. The script processes the input image, identifies coin contours based on circularity and area, and segments each detected coin as a separate image.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- OS module (`os`)

## Installation
Ensure you have Python and the required libraries installed. You can install dependencies using:
```sh
pip install opencv-python numpy
```

## Usage
1. Place the input image in the `input` directory.
2. Run the script:
```sh
python coins.py
```
3. Enter the filename of the image when prompted.
4. Processed images, segmented coins, and output images will be saved in the `output` directory.

## Output
The script generates:
- `<input>_preprocessed.png` - Thresholded and processed image.
- `<inout>_contours.png` - Image with detected contours drawn.
- `<input>_coinX.png` - Individual segmented coin images.
- A console message displaying the number of detected coins.

## Notes
- The script removes existing files in the `output` directory before running.
- Ensure the image has good contrast for accurate detection.

## Example Run
```sh
Enter the name of the image file: coins1.png
Number of coins detected: 2
```

# Part 2


## Overview  
This project performs image stitching using OpenCV. It loads multiple images, detects keypoints, aligns them, and stitches them into a single panoramic image. The final result is saved after optional cropping to remove black borders.  

## Requirements  
- Python 3.x  
- OpenCV (`cv2`)  
- NumPy (`numpy`)  
- Matplotlib (`matplotlib`)  
- Imutils (`imutils`)  
- OS module (`os`)  

## Installation  
Ensure you have Python and the required libraries installed. You can install dependencies using:  
```sh
pip install opencv-python numpy matplotlib imutils

```
## Usage
- Place the input images in a directory.
- The directory part2 has two codes, `panaroma.py` and `panaroma_manual.py`.
- Steps to run `panaroma.py`:
  1. Run the script:
  ```sh
  python panaroma.py
  ```
  2. Enter the path to the directory containing the input images when prompted.
  3. The final stitched image will be saved in the `output` directory.
- Steps to run `panaroma_manual.py`:
    1. Run the script:
    ```sh
    python panaroma_manual.py
    ```
    2. Enter the path to the directory containing the input images when prompted.
    3. The final stitched image will be saved in the `output` directory.

## Notes
- The script removes existing files in the `output` directory before running.
- Ensure the input images have overlapping regions for accurate stitching.


## Documentation
The report for this assignment can be found [here](report.pdf).