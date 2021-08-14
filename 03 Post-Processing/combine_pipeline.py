import numpy as np
from PIL import Image
from pathlib import Path
import os, os.path
import cv2 as cv

# Set thresholds
threshold = 127
blob = 200

# Load grayscale image
greyscale_path = "lidar_predicted_masks_FULL"

########### PROBABILITY FILTERING ###############
output_path_threshold = f"all_masks_t_{threshold}"
if os.path.isdir(output_path_threshold):
    print("Folder already exists")
else:
    print("Creating new folder for masks")
    Path(output_path_threshold).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(greyscale_path):
        img = Image.open(greyscale_path + "/" + f)
        img = np.array(img)
        img[img > threshold] = 255
        img[img <= threshold] = 0
        img_out = Image.fromarray(img)
        img_out.save(output_path_threshold + "/" +  f)
####################################################


########### BLOB FILTERING ###############
output_path_blob =f"all_masks_t_{threshold}_min_{blob}"
if os.path.isdir(output_path_blob):
    print("Blob Folder already exists")
else:
    print("Creating new blob folder")
    Path(output_path_blob).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(output_path_threshold):
        img_mask = cv.imread(output_path_threshold + "/" +f)
        imgray = cv.cvtColor(img_mask, cv.COLOR_BGR2GRAY)
        imgray_float = imgray.astype(float)
        imgray_flip = abs(imgray_float - 255).astype(np.uint8)
        contours, _ = cv.findContours(imgray_flip, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cimg = np.zeros_like(imgray_flip)
            cv.drawContours(cimg, contours, i, color=255, thickness=-1)
            pts = np.where(cimg == 255)
            if pts[0].shape[0]:
                area = int(cv.contourArea(contours[i]))
                if area < blob:
                    img_mask[pts]=255
        cv.imwrite(output_path_blob +"/" + f, img_mask[:,:,1])
####################################################