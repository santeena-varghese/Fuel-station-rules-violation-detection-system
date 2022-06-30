# import the opencv library
import cv2
from newsvm import *
import imutils

# define a video capture object
vid = cv2.VideoCapture("video11.mp4")


while(True):
	
	# Capture the video frame
	# by frame
	ret, frame = vid.read()
	frame = imutils.resize(frame, width=700)
	# Display the resulting frame
	cv2.imshow('frame', frame)
	cv2.imwrite('frame.jpg',frame)
	predictsvm("frame.jpg")
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
