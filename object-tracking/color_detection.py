import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

def color_detection(image, filename):
	img = cv2.imread(image)   # you can read in images with opencv
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# lower boundary RED color range values; Hue (0 - 10)
	lower1 = np.array([0, 100, 20])
	upper1 = np.array([10, 255, 255])
	
	# upper boundary RED color range values; Hue (160 - 180)
	lower2 = np.array([160,100,20])
	upper2 = np.array([179,255,255])
	
	# get mask based on RED color range
	lower_mask = cv2.inRange(hsv_img, lower1, upper1)
	upper_mask = cv2.inRange(hsv_img, lower2, upper2)
	mask = lower_mask + upper_mask

	# find percent coverage
	ratio = cv2.countNonZero(mask)/(hsv_img.size/3)
	percentage = np.round(ratio*100, 2)

	# Plot the coverage
	fig, ax = plt.subplots(1, 1)
	ax.patch.set_edgecolor('black')  
	ax.patch.set_linewidth('1')
	plt.yticks([])
	plt.xticks([])
	plt.title('Pixel percentage: {}%'.format(percentage))
	plt.imshow(mask, cmap='gray')   # this colormap will display in black / white
	
	# Saving the plot
	pltname = 'coverages/{}.jpg'.format(filename)
	plt.savefig(pltname)