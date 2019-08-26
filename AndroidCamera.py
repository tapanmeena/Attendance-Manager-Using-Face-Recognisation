import urllib.request
import cv2
import numpy as np
import time

url='http://192.168.43.198:8080/shot.jpg'

while True:
    # Use urllib to get the image from the IP camera
    imgResp = urllib.request.urlopen(url)
    # Numpy to convert into a array
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    # Finally decode the array to OpenCV usable format ;) 
    img = cv2.imdecode(imgNp,-1)
	# put the image on screen
    cv2.imshow('IPWebcam',img)
    #To give the processor some less stress
    #time.sleep(0.1) 
    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break