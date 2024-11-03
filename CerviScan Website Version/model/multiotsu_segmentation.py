import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_ubyte
from skimage.filters import threshold_multiotsu

def multiotsu_masking(image):
    # Read Image
    img = io.imread(image)

    # Compute multi-Otsu thresholds
    threshold = threshold_multiotsu(img, classes=5)

    # Digitize (segment) the image based on the thresholds
    regions = np.digitize(img, bins=threshold)

    # Convert regions to uint8 explicitly to avoid the warning
    output = (regions * (255 // (regions.max() + 1))).astype(np.uint8)
    output[output < np.unique(output)[-1]] = 0
    output[output >= np.unique(output)[-1]] = 1

    return output