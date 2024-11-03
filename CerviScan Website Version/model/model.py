from rgb_to_gray import rgb_to_gray_converter
from multiotsu_segmentation import multiotsu_masking
from bitwise_operation import get_segmented_image
from feature_extraction.main import get_features

import os

def getPrediction(filename):
    static_path = "../static/image_uploads/"+filename[:-4]
    
    if not os.path.exists(static_path):
        os.makedirs(static_path)
    
    
    