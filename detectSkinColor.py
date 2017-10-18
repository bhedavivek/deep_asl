import numpy as np
import cv2
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def removeface(img, lower, upper):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	low, high = [], []
	for (x,y,w,h) in faces:
		tempImage = img[y:y+h,x:x+w]
		# cv2.imshow('temp',tempImage)
		# cv2.waitKey(0)
		tempImage = cv2.cvtColor(tempImage, cv2.COLOR_BGR2YCR_CB)
		imgShape = tempImage.shape
		reshapedImg = tempImage.reshape((imgShape[0]*imgShape[1], imgShape[2]))
		low.append(np.percentile(reshapedImg,lower, axis=0))
		high.append(np.percentile(reshapedImg,upper, axis=0))

		cv2.rectangle(img,(x,y),(int((x+w)*1.1),int((y+h)*1.1)),(0,0,0), thickness= -1)
	return img, np.mean(low, axis=0), np.mean(high, axis=0)
