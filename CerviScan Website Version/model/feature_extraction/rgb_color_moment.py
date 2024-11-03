import cv2 as cv
import numpy as np
from scipy.stats import skew
import glob
from tqdm import tqdm
import os
import pandas as pd

def get_rgb_color_moment(image_path):
    image = cv.imread(image_path)
    image_flat = image.flatten()

    mean = np.mean(image, dtype=np.float64)
    mean = round(mean, 3)

    skewness = skew(image_flat, axis=0)
    skewness = round(skewness, 3)

    var = np.var(image, dtype=np.float32, ddof=0)
    std = pow(var, 0.5)
    std = round(std, 3)
    
    return [mean, skewness, std]

def get_feature_name():
    return ["mean", "skewness", "std"]

def main(): 
    # Image Folder Path
    folder_path = "../segmented_image/fcm_segmentation"

    # Array Of Image
    image_files = glob.glob(os.path.join(folder_path, '*'))

    # Total number of images
    total_images = len(image_files)

    # TAMURA FEATURES RESULT
    color_moment_name = get_feature_name()
    color_moment_features = []

    # IMAGE NAME
    image_name = []

    for image_file in tqdm(image_files, desc="RGB Color Moment Extraction", unit="file", ncols=100):
        filename = os.path.basename(image_file)[0:-4]
        image_name.append(filename)
        
        image_features = get_rgb_color_moment(image_file)
        color_moment_features.append(image_features)
    
    # Create Data Frame
    df = pd.DataFrame(color_moment_features, columns=color_moment_name)
    df.insert(0, 'Image', image_name)
    
    # Write CSV
    csv_file = '../data/fcm/rgb_color_moment.csv'
    df.to_csv(csv_file, index=False)

# main()
