# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from color_detection import color_detection

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = []

# if a video path was not supplied, grab the reference
# to the webcam
filename = args.get("video", False)
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# Get the Default resolutions
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))

# Define the codec and filename
outfile = 'videos/{}.avi'.format(filename)
out = cv2.VideoWriter(outfile,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# allow the camera or video file to warm up
time.sleep(1.0)

final_frame = None
# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# set up overlay
	cv2.imwrite("frame.jpg", frame)
	overlay = cv2.imread("frame.jpg")
	output = cv2.imread("frame.jpg")

	# resize the frame, blur it, and convert it to the HSV
	# color space
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing rectangle
		c = max(cnts, key=cv2.contourArea)

		# get coordinates of angled rectangle and add to
		# list
		rect = cv2.minAreaRect(c)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		pts.append(box)
		cv2.drawContours(frame,[box],-1,(0,255,255),2)

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# draw contour onto frame
		cv2.drawContours(overlay,[pts[i]],-1,(0,0,255),-1)
		cv2.drawContours(frame,[pts[i]],-1,(0,0,255),-1)

	# apply the overlay
	alpha = 0.3
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	
	# show the frame to our screen
	# cv2.imshow("Frame", frame)
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# write the  frame
	out.write(output)
	final_frame = frame

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# write final frame
pathfile = 'paths/{}.jpg'.format(filename)
cv2.imwrite(pathfile, final_frame)

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
out.release()
cv2.destroyAllWindows()

# run coverage percentages
color_detection(pathfile, filename)