import os
import cv2
for x in os.listdir('veg/'):
   img=cv2.imread("veg/"+x)
   img=cv2.resize(img, (100, 100))
   cv2.imwrite("veg/"+x,img)
   x