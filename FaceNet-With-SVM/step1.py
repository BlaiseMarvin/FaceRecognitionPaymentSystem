import os
from os import listdir
from numpy import asarray
import cv2

from numpy import savez_compressed

def resize_face(filename,required_size=(160,160)):
	face=cv2.imread(filename)
	face=cv2.resize(face,required_size)
	return face

def load_faces(directory):
	faces=list()
	#enumerate files in the list
	for filename in listdir(directory):
		#path
		path=directory+filename
		#resize face
		face=resize_face(path)
		#store
		faces.append(face)
	return faces

def load_dataset(directory):
	X,y=list(),list()
	#enumerate folders, on per class
	for subdir in listdir(directory):
		#path
		path=directory+subdir+'/'
		#load all faces in the sub directory
		faces=load_faces(path)
		#create labels
		labels=[subdir for _ in range(len(faces))]

		#summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		#store
		X.extend(faces)
		y.extend(labels)
	return asarray(X),asarray(y)

# load train dataset
trainX, trainy = load_dataset("E:/FINAL-YEAR-PROJECT/models/trainedFacenetModels/retrained-413to426-Layers/dataset/train/")
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset("E:/FINAL-YEAR-PROJECT/models/trainedFacenetModels/retrained-413to426-Layers/dataset/test/")
print(testX.shape, testy.shape)
# save arrays to one file in compressed format
savez_compressed('RemontadaDataset.npz', trainX, trainy, testX, testy)