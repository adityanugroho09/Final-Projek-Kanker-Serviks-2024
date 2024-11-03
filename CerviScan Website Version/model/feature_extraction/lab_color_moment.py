import numpy as np
from scipy.stats import skew

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import os
import pandas as pd
import cv2
import skimage

def get_lab_color_moment(image_path):    
    image = cv2.imread(image_path)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_array = np.array(rgb_image)
    
    rgb_img_normalized = [[[element / 255 for element in sublist]for sublist in inner_list]for inner_list in image_array]
    
    lab_image = skimage.color.rgb2lab(rgb_img_normalized)

    # Mean
    mean_l = np.mean(lab_image[:, :, 0])
    mean_a = np.mean(lab_image[:, :, 1])
    mean_b = np.mean(lab_image[:, :, 2])
            
    # Std
    std_l = np.std(lab_image[:, :, 0])
    std_a = np.std(lab_image[:, :, 1])
    std_b = np.std(lab_image[:, :, 2])

    # Skewness
    skew_l = skew(lab_image[:, :, 0].flatten())
    skew_a = skew(lab_image[:, :, 1].flatten())
    skew_b = skew(lab_image[:, :, 2].flatten())
    
    return [mean_l, mean_a, mean_b, std_l, std_a, std_b, skew_l, skew_a, skew_b]

def get_feature_name():
    return ['mean_l', 'mean_a', 'mean_b', 'std_l', 'std_a', 'std_b', 'skew_l', 'skew_a', 'skew_b',]

def main(): 
    # Image Folder Path
    folder_path = "../segmented_image/multiotsu"

    # Array Of Image
    image_files = glob.glob(os.path.join(folder_path, '*'))

    # Total number of images
    total_images = len(image_files)

    # TAMURA FEATURES RESULT
    color_moment_name = get_feature_name()
    color_moment_features = []

    # IMAGE NAME
    image_name = []

    for image_file in tqdm(image_files, desc="Color Moment Extraction", unit="file", ncols=100):
        filename = os.path.basename(image_file)[0:-4]
        image_name.append(filename)
        
        image_features = get_lab_color_moment(image_file)
        color_moment_features.append(image_features)
    
    # Create Data Frame
    df = pd.DataFrame(color_moment_features, columns=color_moment_name)
    df.insert(0, 'Image', image_name)
    
    # Write CSV
    csv_file = '../data/multiotsu/lab_color_moment.csv'
    df.to_csv(csv_file, index=False)

# main()