#!/usr/bin/env python
import numpy as np
import math
import cv2
from time import sleep


color_checker = cv2.imread('color_checker.jpg')

rows, columns, dim = color_checker.shape
scale= 0.2
color_checker = cv2.resize(color_checker, (int(columns*scale), int(rows*scale)))
rows, columns, dim = color_checker.shape

gray_color = cv2.cvtColor(color_checker, cv2.COLOR_BGR2GRAY)
cv2.imshow('', gray_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

canny_color = cv2.Canny(gray_color, 100, 200)
cv2.imshow('Canny', canny_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
