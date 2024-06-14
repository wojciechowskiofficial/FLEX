import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm


root_dir = "/home/adamwsl/MALE/occlusion_study_results_all_3/occlusion_study_sat_results"

for image_folder in tqdm(os.listdir(root_dir)):
    image_folder_contents = os.listdir(os.path.join(root_dir, image_folder))
    for file in image_folder_contents:
        if file.endswith(".JPEG"): og_image = plt.imread(os.path.join(root_dir, image_folder, file))
        if file.endswith(".jpg") and not file.endswith("_viz.jpg"): blocked_image = plt.imread(os.path.join(root_dir, image_folder, file))
    
    blocked_image_gray = cv2.cvtColor(blocked_image, cv2.COLOR_BGR2GRAY)
    mask = np.where(blocked_image_gray == 255, blocked_image_gray, 0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((13, 13), dtype=np.uint8)
    dilated_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_DILATE, kernel)
    masked_image = np.where(np.dstack([dilated_mask] * 3) == 255, og_image, 0)
    
    plt.imsave(os.path.join(root_dir, image_folder, image_folder + "masked_negated.jpg"), masked_image)