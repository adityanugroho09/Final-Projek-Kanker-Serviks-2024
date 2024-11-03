import os
import glob
from tqdm import tqdm
import pandas as pd
from pprint import pprint

from lbp_extraction import get_lbp_features, get_lbp_name
from tamura_features import get_tamura, get_tamura_name
from glrlm_extraction import get_glrlm, get_glrlm_name
from lab_color_moment import get_lab_color_moment, get_feature_name as lab_color_moment_name

features_dictionary = {
    'lbp': {
        'feature_extractor': get_lbp_features,
        'features_name': get_lbp_name(),
        'features': []
        },
    'tamura': {
        'feature_extractor': get_tamura,
        'features_name': get_tamura_name(),
        'features': []
        },
    'glrlm': {
        'feature_extractor': get_glrlm,
        'features_name': get_glrlm_name(),
        'features': []
        },
    'lab_color_moment': {
        'feature_extractor': get_lab_color_moment,
        'features_name': lab_color_moment_name(),
        'features': []
        },
    }

def get_features(image):
    for feature_extraction_method in features_dictionary:
        feature_extractor = features_dictionary[feature_extraction_method]['feature_extractor']
        feature = feature_extractor(image)
        features_dictionary[feature_extraction_method]['features'].append(feature)
            
    all_features = {}

    for key in features_dictionary:
        df = pd.DataFrame(features_dictionary[key]['features'], columns=features_dictionary[key]['features_name'])
        df = df.loc[:, (df != df.iloc[0]).any()]
        all_features[key] = df


    df_lab_color_moment_lbp_glrlm_tamura = pd.concat([
        all_features["lab_color_moment"],
        all_features["lbp"],
        all_features["glrlm"],
        all_features["tamura"],
        ], axis=1)

    return df_lab_color_moment_lbp_glrlm_tamura
