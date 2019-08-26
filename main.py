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
from keras.models import load_model
from keras.layers import Dense,Activation
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
'''
	For disabling warning while importing tensorflow and MTCNN
'''
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #For suppressing warnings.0 = all messages are logged (default behavior),1 = INFO messages are not printed,2 = INFO and WARNING messages are not printed,3 = INFO, WARNING, and ERROR messages are not printed

# import tensorflow as tf
from mtcnn.mtcnn import MTCNN

class emb:
	def __init__(self):
		self.model = load_model('facenet_keras.h5')
	def calculate(self, img):
		return self.model.predict(img)[0]

class DenseArchs:
	def __init__(self, classes):
		print("Training Initiated ")
		self.model = Sequential()
		self.classes = classes

	def arch(self):
		self.model.add(Dense(64,input_dim=128))
		self.model.add(LeakyReLU(alpha=0.1))
		self.model.add(Dense(32))
		self.model.add(LeakyReLU(alpha=0.1))
		self.model.add(Dense(16))
		self.model.add(LeakyReLU(alpha=0.1))
		self.model.add(Dense(self.classes))
		self.model.add(Activation('softmax'))

		return self.model

class face:
	def __init__(self):
	    self.cascade=cv2.CascadeClassifier('faces.xml')
	    self.x=None
	    self.y=None
	    self.w=None
	    self.h=None

	def detectFace(self,img):
	    cropped=None
	    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	    faces=self.cascade.detectMultiScale(grey,1.3,5)
	    cropped=[]
	    coor=[]
	    for (self.x,self.y,self.w,self.h) in faces:
	        cropped.append(img[self.y:self.y+self.h,self.x:self.x+self.w])
	        coor.append([self.x,self.y,self.w,self.h])
	    return cropped,coor

def cut_faces(frames, faces_cords):
	face_no = 1
	for result in faces_cords:
		x, y, w, h = result['box']
		w_rm = int(0.2 * w/2)
		face = frames[y : y + h, x: x + w]
		cv2.imshow('Detected Face', face)
		facePath = 'face/' + str(face_no) + '.jpg'
		# cv2.imwrite(facePath, face)
		cv2.waitKey(10000)
		face_no += 1

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
		# img = cv2.transpose(img, img)
		# img = cv2.flip(img, 1)
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

def add_face_from_video_file(filename):
	person_rollNumber = input("Person Roll Number : ")
	folder = 'people/' + person_rollNumber
	if not os.path.exists(folder):
		os.makedirs(folder)
		input("Press Enter to Record Face ")

		video = cv2.VideoCapture(filename)
		check, frame = video.read()

		detector = MTCNN()
		photosTaken = 1
		while(video.isOpened()):
			check, frame = video.read()
			frame = cv2.transpose(frame, frame)
			frame = cv2.flip(frame, 1)

			if not check:
				break;
			else:
				faces = detector.detect_faces(frame)

				if(len(faces) != 1):
					pass

				for result in faces:
					x, y, w, h = result['box']

					face = frame[y: y+h, x:x+w]
					filePath = folder +'/' + str(photosTaken) + '.jpg'
					cv2.imwrite(filePath, face)
					print("Images Saved : " + str(photosTaken))
					photosTaken += 1
					# cv2.imshow("Face Saved ", face)
					# cv2.waitKey(100)
	else:
		print("Person Record Already Exists")

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

def getFaces(filename):
	detector = MTCNN()
	image = pyplot.imread(filename)
	faces = detector.detect_faces(image)
	cut_faces(image, faces)

def train_model():
	number_of_students = 5
	e = emb()
	arc = DenseArchs(number_of_students)
	face_model = arc.arch()

	x_data = []
	y_data = []

	learning_rate = 0.01
	epochs = 27
	batch_size = 32

	people = os.listdir('people')

	for x in people:
		for i in os.listdir('people/' + x):
			path = 'people/'+x+'/'+i
			print(path)
			img = cv2.imread('people/' + x +'/' + i, 1)
			img = cv2.resize(img, (160,160))
			img = img.astype('float')/255.0
			img = np.expand_dims(img, axis = 0)
			embs = e.calculate(img)
			x_data.append(embs)
			y_data.append(int(x))

	x_data=np.array(x_data,dtype='float')
	y_data=np.array(y_data)
	y_data=y_data.reshape(len(y_data),1)
	x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1,random_state=77)
	y_train=to_categorical(y_train,num_classes=number_of_students)
	y_test=to_categorical(y_test,num_classes=number_of_students)

	o=Adam(lr=learning_rate,decay=learning_rate/epochs)
	face_model.compile(optimizer=o,loss='categorical_crossentropy')
	face_model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,shuffle='true',validation_data=(x_test,y_test))
	face_model.save('face_reco2.MODEL')
	print(x_data.shape,y_data.shape)

def recognise():
	label=None
	a={0:0,1:0, 2:0, 3:0, 4:0, 5:0}
	people={0:"nikhilesh",1:"prabhakar", 2:"tapan", 3:"rishi", 4:"rishu"}
	abhi=None
	#data=database()
	e=emb()
	fd=face()

	print('attendance till now is ')
	#data.view()

	model=load_model('face_reco2.MODEL')

	cap=cv2.VideoCapture(0)
	ret=True
	# test()
	while ret:
	    ret,frame=cap.read()
	    frame=cv2.flip(frame,1)
	    det,coor=fd.detectFace(frame)

	    if(det is not None):
	        for i in range(len(det)):
	            detected=det[i]
	            k=coor[i]
	            f=detected
	            detected=cv2.resize(detected,(160,160))
	            #detected=np.rollaxis(detected,2,0)
	            detected=detected.astype('float')/255.0
	            detected=np.expand_dims(detected,axis=0)
	            feed=e.calculate(detected)
	            feed=np.expand_dims(feed,axis=0)
	            prediction=model.predict(feed)[0]

	            result=int(np.argmax(prediction))
	            if(np.max(prediction)>.70):
	                for i in people:
	                    if(result==i):
	                        label=people[i]
	                        if(a[i]==0):
	                            print("a")
	                        a[i]=1
	                        abhi=i
	            else:
	                label='unknown'
	            #data.update(label)


	            cv2.putText(frame,label,(k[0],k[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
	            # if(abhi is not None):
	            #     if(a[abhi]==1):
	            #         cv2.putText(frame,"your attendance is complete",(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
	            cv2.rectangle(frame,(k[0],k[1]),(k[0]+k[2],k[1]+k[3]),(252,160,39),3)
	            cv2.imshow('onlyFace',f)
	    cv2.imshow('frame',frame)
	    if(cv2.waitKey(1) & 0XFF==ord('q')):
	        break
	cap.release()
	cv2.destroyAllWindows()


def argument_parse():
	parser = argparse.ArgumentParser(description='Attendence Manager using face detection')
	parser.add_argument('-i','--image',help='To process an image file',action='store')
	parser.add_argument('-v','--video',help='To process an video file',action='store')
	parser.add_argument('-c','--live',help='To process live capture',action='store_true')
	parser.add_argument('-t','--train',help='To train the model on image data',action='store_true')
	parser.add_argument('-a','--addFace',help='To Register New Face into Database',action='store_true')
	parser.add_argument('-av','--addFaceVideo',help='To Register New Face from Video file into Database',action='store')
	parser.add_argument('-g','--getFace',help='To Get all faces in a image',action='store')
	parser.add_argument('-p','--predict',help='To Prdict person from live capture',action='store_true')
	
	args = parser.parse_args()
	if args.image:
		image_file=args.image
		find_face_in_image(image_file)
	elif args.getFace:
		image_file = args.getFace
		getFaces(image_file)
	elif args.video:
		video_file=args.video
		find_face_in_video_file(video_file)
	elif args.live:
		live_capture()
	elif args.train:
		train_model()
	elif args.addFace:
		Add_Face()
	elif args.addFaceVideo:
		video_file = args.addFaceVideo
		add_face_from_video_file(video_file)
	elif args.predict:
		recognise()
	else:
		print("No arguments given. Use -h option for list of all arguments available.")

if __name__ == '__main__':
	argument_parse()
	# GenDataSet()