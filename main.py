#libraries
import cv2
import sys
import warnings
import logging
import os
import time
import argparse
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import numpy as np
'''
	For disabling warning while importing tensorflow and MTCNN
'''
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #For suppressing warnings.0 = all messages are logged (default behavior),1 = INFO messages are not printed,2 = INFO and WARNING messages are not printed,3 = INFO, WARNING, and ERROR messages are not printed

# import tensorflow as tf
from mtcnn.mtcnn import MTCNN

#make boxes on detected faces
def make_boxes(filename, result_list):
	data = pyplot.imread(filename)
	pyplot.imshow(data)

	# get the context for drawing boxes
	ax = pyplot.gca()

	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='green')
		# draw the box
		ax.add_patch(rect)

	# show the plot
	# pyplot.show()
	# pyplot.savefig(saveFileName)
	pyplot.show()

# filename = sys.argv[1]
# files = ['testingImage/1.jpg','testingImage/2.jpg','testingImage/3.jpg','testingImage/4.jpg','testingImage/5.jpg','testingImage/6.jpg','testingImage/7.jpg','testingImage/8.jpg','testingImage/9.jpg']
# files = ['10.jpg', '11.jpg']
def find_face_in_image(filename):
	detector = MTCNN()
	# saveFileName = str(filename) + '.jpg'
	# filename = 'testingImage/' + str(filename) +'.jpg'
	image = pyplot.imread(filename)
	faces = detector.detect_faces(image)
	# make_boxes(filename, faces, saveFileName)
	make_boxes(filename, faces)
	# print(saveFileName, " is Done.")

def find_face_in_video_file(filename):
	cam = cv2.VideoCapture(filename)
	ret, img = cam.read()
	detector = MTCNN()

	#to save video
	# fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# output_movie = cv2.VideoWriter('VideoOutput/4.mp4', fourcc, 25, (640, 360))
	while(cam.isOpened()):
		ret, img = cam.read()

		if not ret:
			break
		else:
			faces = detector.detect_faces(img)

			for result in faces:
				x, y, w, h = result['box']
				# create the shape
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

			cv2.imshow("Face",img)
			if(cv2.waitKey(1) & 0xFF==ord('q')):
				break
			# output_movie.write(img)

def live_capture():
	cam = cv2.VideoCapture(0)
	ret, img = cam.read()
	detector = MTCNN()

	while True:
		ret, img = cam.read()

		if not ret:
			break
		else:
			faces = detector.detect_faces(img)

			for result in faces:
				x, y, w, h = result['box']
				# create the shape
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

			cv2.imshow("Face",img)
			if(cv2.waitKey(1) & 0xFF==ord('q')):
				break

def Add_Face():
	person_rollNumber = input("Person Roll Number : ")
	folder = 'people/'+person_rollNumber
	if not os.path.exists(folder):
		os.makedirs(folder)
		input("Press Enter to Record Face ")

		video = cv2.VideoCapture(0)
		# detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		# detector = cv2.CascadeClassifier('faces.xml')
		detector = MTCNN()

		photosTaken = 1

		while photosTaken !=31:
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


def argument_parse():
	parser = argparse.ArgumentParser(description='Attendence Manager using face detection')
	parser.add_argument('-i','--image',help='To process an image file',action='store')
	parser.add_argument('-v','--video',help='To process an video file',action='store')
	parser.add_argument('-c','--live',help='To process live capture',action='store_true')
	parser.add_argument('-t','--train',help='To train the model on image data',action='store_true')
	parser.add_argument('-a','--addFace',help='To Register New Face into Database',action='store_true')
	
	args = parser.parse_args()
	if args.image:
		image_file=args.image
		find_face_in_image(image_file)
	elif args.video:
		video_file=args.video
		find_face_in_video_file(video_file)
	elif args.live:
		live_capture()
	elif args.train:
		train_model()
	elif args.addFace:
		Add_Face()
	else:
		print("No arguments given. Use -h option for list of all arguments available.")

if __name__ == '__main__':
	argument_parse()
	# GenDataSet()