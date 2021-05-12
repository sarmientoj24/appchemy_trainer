import cv2
import glob, os
import pandas as pd
from PIL import Image
import numpy as np
import imageio


def read_imgs_to_np_from_folder(img_folder):
    imgs = glob.glob(img_folder + '*.jpg')

    X = []
    for f in imgs:
        x = imageio.imread(f)

        if len(x.shape) == 3 and x.shape[0] == 260 and x.shape[1] == 195 and x.shape[2] == 3:
            x = x.astype(np.float32) / 255.0
            X.append(x)
    
    X = np.array(X)
    print(X.shape)
    return X