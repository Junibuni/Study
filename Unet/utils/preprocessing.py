import cv2
import numpy as np
import matplotlib.pyplot as plt

def polygon2mask():
    mask2 = np.zeros((100,100),dtype = np.uint8)
    polygon1 = np.array([[5,5],[30,30],[60,10],[70,70],[50,45],[5,80]])
    polygon2 = np.array([[40,80],[45,75],[43,70],[60,80],[50,90]])

    cv2.fillPoly(mask2,[polygon1],1)
    cv2.fillPoly(mask2,[polygon2],2)