import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
import glob
import os
images = []
labels = []
paths = []
# i = 0
for path in glob.glob("/Users/vineet/Desktop/cnn/English/Img/BadImag/Bmp/Sample*"):
#     print(path)
#     for img in glob.glob("/Users/vineet/Desktop/cnn/English/img/GoodImg/Bmp/Sample001/*.png"):
    for img in glob.glob(path+"/*.png"):
        paths.append(img)
        os.rename(img, img + ".png") 
#         n= plt.imread(img)
        # images.append(np.resize(n,(32,32,3)))
        # prefix = img.split("/Sample")
        # prefix = prefix[1].split("/")
        # labels.append(prefix[0])