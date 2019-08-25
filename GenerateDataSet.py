import cv2, os, time
import numpy as np
from mtcnn.mtcnn import MTCNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import logging
import warnings
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

number_of_person = int(input("Number of Persons: "))

for person in range(number_of_person):
	person_rollNumber = input("Person Roll Number : ")
	folder = 'people/'+person_rollNumber
	if not os.path.exists(folder):
		os.makedirs(folder)
		input("Press Enter to Record Face ")

		video = cv2.VideoCapture(0)
		# detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		# detector = cv2.CascadeClassifier('faces.xml')
		detector = MTCNN()

		photosTaken = 0

		while photosTaken !=210:
			check, frame = video.read()
			faces = detector.detect_faces(frame)

			if(len(faces) != 1):
				pass

			for result in faces:
				x, y, w, h = result['box']

				face = frame[y:y+h, x:x+w]
				# create the shape
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255,0), 2)
				filePath = folder +'/' + str(photosTaken) + '.jpg'
				cv2.imwrite(filePath, face)
				print("Images Saved : " + str(photosTaken))
		        # cv2.imwrite(name+'/'+str(pic_no)+'.jpg',cropped)
				photosTaken += 1
				cv2.imshow('Saved Face', face)
			cv2.imshow('Face ', frame)
			cv2.waitKey(100)


		video.release()
		cv2.destroyAllWindows()
	else:
		print("Person Record Already Exists")



'''
	important but Unused Code
	Don't Delete it for Now
'''
# import cv2, time
# import matplotlib.pyplot as plt
# data = plt.imread('testingImage/2.jpg')
# plt.imshow(data)
# plt.show()
# time.sleep(400)

# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# faces = detector.detectMultiScale(gray, 1.3, 5)

# for (x,y,w,h) in faces:
# 	face = frame[y:y+h, x:x+w]
# 	cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2, cv2.LINE_AA)
# 	cv2.imwrite(person_rollNumber + '/' + str(photosTaken)+'.jpg', face)
# 	photosTaken += 1
# cv2.imshow('Face ',frame)
# cv2.waitKey(100)
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# faces = detector.detectMultiScale(gray, 1.3, 5)

# for(x, y, w, h) in faces:
# 	face = frame[y: y+h, x:w+h]
# 	cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), cv2.LINE_AA)
# 	photosTaken += 1
# 	filePath = folder +'/' + str(photosTaken) + '.jpg'
# 	cv2.imwrite(filePath, face)
# cv2.imshow('Frame', frame)
# cv2.waitKey(100)
