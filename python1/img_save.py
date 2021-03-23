import numpy as np
import cv2

c = np.load('geekfile.npy')
fp = "2021_2_img.bmp"


# put name of the file here
cv2.imwrite("2021_3.jpg",c)
# print(c)
