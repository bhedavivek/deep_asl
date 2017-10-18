import cv2
import numpy
from detectSkinColor import removeface

def nothing():
	return

def find_hand(img):

	gestureStatic = 0
	gestureDetected = 0

	# previous frame contour of hand. Used to compare with new contour to find if gesture has changed.
	prevcnt = numpy.array([], dtype=numpy.int32)

	# previous values of cropped variable
	x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0

	sourceImage = img
	cv2.createTrackbar('lower','image',0,100,nothing)
	cv2.createTrackbar('upper','image',0,100,nothing)

	# while True:

	sourceImage, min_YCrCb, max_YCrCb = removeface(sourceImage, 25,95)

	min_YCrCb, max_YCrCb = numpy.array((0, 138, 67)), numpy.array((255, 173, 133))

	# Convert image to YCrCb
	imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)
	# imageYCrCb = cv2.GaussianBlur(imageYCrCb, (5, 5), 0)

	# Find region with skin tone in YCrCb image
	skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

	# Do contour detection on skin region
	_, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) > 0 :
		# sorting contours by area. Largest area first.
		contours = sorted(contours, key=cv2.contourArea, reverse=True)

		# get largest contour and compare with largest contour from previous frame.
		# set previous contour to this one after comparison.
		cnt = contours[0]
		prevcnt = contours[0]

		# once we get contour, extract it without background into a new window called handTrainImage
		stencil = numpy.zeros(sourceImage.shape).astype(sourceImage.dtype)
		color = [255, 255, 255]
		cv2.fillPoly(stencil, [cnt], color)
		handTrainImage = cv2.bitwise_and(sourceImage, stencil)

		# crop coordinates for hand.
		x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)
		if w_crop < h_crop:
			x_crop = x_crop - (h_crop - w_crop)/2
			w_crop = h_crop
		else:
			y_crop = y_crop - (w_crop - h_crop)/2
			h_crop = w_crop

		# Training image with black background
		handTrainImage = handTrainImage[y_crop -15 : y_crop + h_crop + 15, x_crop - 15 : x_crop + w_crop + 15]

		# drawing the largest contour
		cv2.drawContours(sourceImage, contours, 0, (0, 255, 0), 1)

		if handTrainImage.shape[0] > 10 and handTrainImage.shape[1] > 10:
			handTrainImage = cv2.resize(handTrainImage, (200,200))
		
		return cnt, handTrainImage
	else:
		return None, None
