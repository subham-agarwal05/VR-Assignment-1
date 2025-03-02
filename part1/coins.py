import cv2
import numpy as np
import os

def display_image(img, title="Image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def save_image(img,name="output.png"):
    output_image_path = "output"
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)
    cv2.imwrite(os.path.join(output_image_path, name), img)
    
def count_coins(coins):
    #count the number of coins
    print("Number of coins detected: ", len(coins))
    
def segment_coins(img,contours,input_image_path):
    coins=[]
    #segment each coin from the image
    for i,cnt in enumerate(contours):
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        coin = cv2.bitwise_and(img, img, mask=mask)
        x, y, w, h = cv2.boundingRect(cnt)
        coin = coin[y:y+h, x:x+w]
        display_image(coin, "Coin "+str(i+1))
        name = os.path.basename(input_image_path).split(".")[0]
        save_image(coin, name+"_coin"+str(i+1)+".png")
        cv2.waitKey(0)
        coins.append(coin)
    cv2.destroyAllWindows()
    return coins
    
    
def draw_contours(img,thresh,scale,input_image_path):
    contours,_= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if (perimeter := cv2.arcLength(cnt, True)) > 0 
                         and 0.7 < (circularity := 4 * np.pi * (cv2.contourArea(cnt) / (perimeter ** 2))) < 1.2
                         and cv2.contourArea(cnt) > 500 * (scale ** 2)]
    #copy the image to draw contours on it
    img_copy = img.copy()
    cv2.drawContours(img_copy, filtered_contours, -1, (0, 255, 0), 2)
    display_image(img_copy, "Contours")
    name = os.path.basename(input_image_path).split(".")[0]
    save_image(img_copy, name+"_contours.png")
    return filtered_contours
    

def preprocess_image(input_image_path):
    #load image
    img = cv2.imread(input_image_path)
    #resising all images to have mazimum dimension of 600pixels
    scale=600/max(img.shape[:2])
    img=cv2.resize(img,None,fx=scale,fy=scale)
    #convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #contrast stretching
    min_val = np.min(gray)
    max_val = np.max(gray)
    stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    #apply median blur to the image
    blur = cv2.medianBlur(stretched, 11)
    #apply thresholding to the image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    display_image(thresh, "Preprocessed Image")
    name = os.path.basename(input_image_path).split(".")[0]
    save_image(thresh, name+"_preprocessed.png")
    return img,thresh,scale
    

def main(path="coins1.png"):
    
    img,thresh,scale=preprocess_image(path)
    contours= draw_contours(img,thresh,scale,path)
    segmented_coins=segment_coins(img,contours,path)
    count_coins(segmented_coins)
    
    
    
if __name__ == "__main__":
    #path for input and output directory
    input_path = "input"
    output_path = "output"
    
    #create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    #if output directroy is not empty, delete all files in it
    for file in os.listdir(output_path):
        os.remove(os.path.join(output_path, file))
        
    #ask user for input image
    input_image = input("Enter the name of the image file: ")
    main(os.path.join(input_path, input_image))
