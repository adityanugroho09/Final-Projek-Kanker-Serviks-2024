import os
import numpy as np
import glob
import warnings
import pandas as pd
from tqdm import tqdm
from GrayRumatrix import getGrayRumatrix

warnings.filterwarnings("ignore")

def get_glrlm_features_name(features, degs):
    glrlm_features_name = []
    for deg in degs:
        for feature in features:
            glrlm_features_name.append(f"{feature}_{deg[0]}")
    return glrlm_features_name

def get_glrlm_name():
    # GLRLM
    glrlm_features = ['SRE', 'LRE', 'GLN', 'RLN', 'RP', 'LGLRE', 'HGL', 'SRLGLE', 'SRHGLE', 'LRLGLE', 'LRHGLE']
    glrlm_degs = [['deg0'], ['deg45'], ['deg90'], ['deg135']]
    glrlm_features_name = get_glrlm_features_name(glrlm_features, glrlm_degs)
    
    return glrlm_features_name

def get_glrlm(path, lbp='off'):
    
    test = getGrayRumatrix()
    test.read_img(path, lbp)

    DEG = [['deg0'], ['deg45'], ['deg90'], ['deg135']]

    glrlm_features_value = []

    for deg in DEG:
        test_data = test.getGrayLevelRumatrix(test.data,deg)
        
        #1
        SRE = test.getShortRunEmphasis(test_data) 
        SRE = np.squeeze(SRE)
        SRE = float(SRE)
        
        #2
        LRE = test.getLongRunEmphasis(test_data)
        LRE = np.squeeze(LRE)
        LRE = float(LRE)
        
        #3
        GLN = test.getGrayLevelNonUniformity(test_data)
        GLN = np.squeeze(GLN)
        GLN = float(GLN)
        
        #4
        RLN = test.getRunLengthNonUniformity(test_data)
        RLN = np.squeeze(RLN)
        RLN = float(RLN)

        #5
        RP = test.getRunPercentage(test_data)
        RP = np.squeeze(RP)
        RP = float(RP)
        
        #6
        LGLRE = test.getLowGrayLevelRunEmphasis(test_data)
        LGLRE = np.squeeze(LGLRE)
        LGLRE = float(LGLRE)
        
        #7
        HGL = test.getHighGrayLevelRunEmphais(test_data)
        HGL = np.squeeze(HGL)
        HGL = float(HGL)
        
        #8
        SRLGLE = test.getShortRunLowGrayLevelEmphasis(test_data)
        SRLGLE = np.squeeze(SRLGLE)
        SRLGLE = float(SRLGLE)
        
        #9
        SRHGLE = test.getShortRunHighGrayLevelEmphasis(test_data)
        SRHGLE = np.squeeze(SRHGLE)
        SRHGLE = float(SRHGLE)
        
        #10
        LRLGLE = test.getLongRunLow(test_data)
        LRLGLE = np.squeeze(LRLGLE)
        LGLRE = float(LGLRE)
        
        #11
        LRHGLE = test.getLongRunHighGrayLevelEmphais(test_data)
        LRHGLE = np.squeeze(LRHGLE)
        LRHGLE = float(LRHGLE)

        glrlm_features_value_per_deg = [SRE, LRE, GLN, RLN, RP, LGLRE, HGL, SRLGLE, SRHGLE, LRLGLE, LRHGLE]
        
        for value in glrlm_features_value_per_deg:
            glrlm_features_value.append(value)

    return glrlm_features_value 

def get_glrlm_on(path):
    return get_glrlm(path, lbp='on')

def main(): 
    # Image Folder Path
    folder_path = "../segmented_image/Normal_multiotsuresult"

    # Array Of Image
    image_files = glob.glob(os.path.join(folder_path, '*'))

    # Total number of images
    total_images = len(image_files)

    # TAMURA FEATURES RESULT
    glrlm_features_name = get_glrlm_name()
    glrlm_features = []

    # IMAGE NAME
    image_name = []

    for image_file in tqdm(image_files, desc="GLRLM Extraction", unit="file", ncols=100):
        filename = os.path.basename(image_file)[0:-4]
        image_name.append(filename)
        
        image_features = get_glrlm(image_file, "off")
        glrlm_features.append(image_features)
    
    # Create Data Frame
    df = pd.DataFrame(glrlm_features, columns=glrlm_features_name)
    df.insert(0, 'Image', image_name)
    
    # Write CSV
    csv_file = '../data/normal_multiotsu/glrlm.csv'
    df.to_csv(csv_file, index=False)

# main()
