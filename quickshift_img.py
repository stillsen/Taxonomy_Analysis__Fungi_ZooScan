import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import skimage
import skimage.segmentation
import cv2

# img_fun = mpimg.imread('/home/stillsen/Documents/Data/Image_classification_soil_fungi__working_copy/cuts/ml2/Ascomycota/Eka_9.12.17_C__9.12.17_C__cut__5.png')
# img_fun = mpimg.imread('/home/stillsen/Documents/Data/Image_classification_soil_fungi__working_copy/cuts/ml2/Ascomycota/Eka_9.12.17_M__9.12.17_M__cut__5.png')
# img_fun = mpimg.imread('/home/stillsen/Documents/Data/Image_classification_soil_fungi__working_copy/cuts/ml2/Ascomycota/MOM_EX_010_B__27.02.18_B__cut__0.png')
# img_fun = mpimg.imread('/home/stillsen/Documents/Data/Image_classification_soil_fungi__working_copy/cuts/ml2/Ascomycota/MOM_EX_012_B__1.03.18_B__cut__5.png')


img = mpimg.imread('/home/stillsen/Documents/Data/ZooNet/ZooScanSet/imgs/Cavoliniidae/993787.jpg')
# img = mpimg.imread('/home/stillsen/Documents/Data/ZooNet/ZooScanSet/imgs/Chaetognatha/992745.jpg')

color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# plt.imshow(color_img)

superpixels = skimage.segmentation.quickshift(color_img, kernel_size=4,max_dist=200, ratio=0.2)
num_superpixels = np.unique(superpixels).shape[0]
skimage.io.imshow(skimage.segmentation.mark_boundaries(img/2+0.5, superpixels))


plt.show()