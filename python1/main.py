#!/usr/bin/python

import numpy as np

from scipy.io import loadmat
from matplotlib.pyplot import imread

import pylab as pl

from algorithm import GEFolki, EFolki, Folki
from tools import wrapData
import cv2
from PIL import Image
# from PIL import save
import matplotlib.pyplot as plt


def demo():
    print("Debut recalage Lidar/Radar\n")
    img=Image.open('2016.bmp')
    limg=Image.open('Yamaguchi4_S4R_RGB1.bmp')
    wt,ht= limg.size
    width, height = img.size
    print("2016",width,height)
    print("2020",wt,ht)

    #img = img.resize((500,1224))
    #limg=limg.resize((500,1224))
    #img.save('../datasets/2016.bmp')
    #limg.save('../datasets/2020.bmp')
    radar = imread('2016.bmp')
    Ilidari = imread('Yamaguchi4_S4R_RGB1.bmp')

    # pl.figure()
    # pl.imshow(radar)
    # pl.title('Radar in pauli color')

    # pl.figure()
    # pl.imshow(Ilidari)
    # pl.title('Lidar in colormap jet')
    #radar1=radar[:,:,0]
    #Ilidari1=Ilidari[:,:,1]
    Iradar = radar[:,:,0]
    Iradar = Iradar.astype(np.float32)/255
    #Ilidari=Ilidari1
    Ilidar = Ilidari.astype(np.float32)/255
    Ilidar=Ilidar[:,:,1]
    print("I",Ilidar.shape)
    print("R",Iradar.shape)
    # u, v = EFolki(Iradar, Ilidar, iteration=2, radius=[32, 24, 16, 8], rank=4, levels=5)
    # N = np.sqrt(u**2+v**2)
    # pl.figure()
    # pl.imshow(N)
    # pl.title('Norm of LIDAR to RADAR registration')
    # pl.colorbar()

    # Ilidar_resampled = wrapData(Ilidar, u, v)

    # C = np.dstack((Ilidar, Iradar, Ilidar))
    # pl.figure()
    # pl.imshow(C)
    # pl.title('Imfuse of RADAR and LIDAR')

    # D = np.dstack((Ilidar_resampled, Iradar, Ilidar_resampled))
    # pl.figure()
    # pl.imshow(D)
    # pl.title('Imfuse of RADAR and LIDAR after coregistration')

    # pl.figure()
    # pl.imshow(Ilidar_resampled)
    # pl.title('Ilidar_resampled')




    print("Fin recalage Lidar/Radar \n\n")

    print("Debut recalage optique/Radar\n")
    #radar = imread(imgs)
    #Ioptique = imread(limg)

    # pl.figure()
    # pl.imshow(radar)
    # pl.title('Radar in pauli color')

    # pl.figure()
    # pl.imshow(Ilidari)
    # pl.title('Optique')

    Iradar = radar[:,:,0]
    Iradar = Iradar.astype(np.float32)
    Ioptique = Ilidari[:,:,1]
    Ioptique = Ioptique.astype(np.float32)
    print(Ioptique.shape)
    print(Iradar.shape)
    u, v = GEFolki(Iradar, Ioptique, iteration=2, radius=range(32, 4, -4), rank=4, levels=6)

    N = np.sqrt(u**2+v**2)
    # pl.figure()
    # pl.imshow(N)
    # pl.title('Norm of OPTIC to RADAR registration')
    # pl.colorbar()

    Ioptique_resampled = wrapData(Ioptique, u, v)

    # C = np.dstack((Ioptique/255, Iradar/255, Ioptique/255))
    # pl.figure()
    # pl.imshow(C)
    # pl.title('Imfuse of RADAR and OPTIC')

    D = np.dstack((Ioptique_resampled/255, Iradar/255, Ioptique_resampled/255))
    # pl.figure()
    # pl.imshow(D)
    # pl.title('Imfuse of RADAR and OPTIC after coregistration')
    # print("Fin recalage optique/Radar \n\n")
    # figure = plt.gcf()
    # pl.figure()
    # pl.imshow(Ioptique_resampled/255)
    # pl.title('Ioptique_resampled')
    # cv2.imwrite("2021.bmp",Ioptique_resampled/255)
    # plt.savefig("myplot.png", dpi = 100)
    print(Ioptique_resampled.shape)
    fp = "2021_2_img.bmp"
    with open(fp, 'w') as f:
        result.save(Ioptique_resampled,f)
    # np.save('geekfile', Ioptique_resampled)
    # Ioptique_resampled.save("geeks.jpg")
    # pl.figure()
    # pl.imshow(Iradar/255)
    # pl.title('Iradar')

if __name__ == '__main__':
    demo()
    pl.show()
else:
    pl.interactive(True)
    demo()
