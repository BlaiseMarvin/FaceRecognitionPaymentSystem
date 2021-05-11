from numpy import load
import cv2
import os
import numpy as np 
from numpy import asarray
from numpy import savez_compressed
from inference import Network

model="E:/FINAL-YEAR-PROJECT/models/trainedFacenetModels/retrained-413to426-Layers/facenet-413426-LastLayer.xml"

def preprocessing(input_image,height,width):
	preprocessed_image=np.copy(input_image)
	preprocessed_image=cv2.resize(preprocessed_image,(width,height))
	preprocessed_image=preprocessed_image.transpose((2,0,1))
	preprocessed_image=preprocessed_image.reshape(1,3,height,width)
	return preprocessed_image

def get_embeddings(model,face_pixels):
	face_pixels=face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std

	plugin=Network()
	plugin.load_model(model=model)
	b,c,h,w=plugin.get_input_shape()
	preprocessed_image=preprocessing(face_pixels,h,w)
	plugin.async_inference(preprocessed_image)
	status=plugin.wait()
	if status==0:
		embz=plugin.extract_output()
		return embz[0]

data=load('RemontadaDataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)


newTrainX=list()

for face_pixels in trainX:
	embedding=get_embeddings(model,face_pixels)
	newTrainX.append(embedding)

newTrainX = asarray(newTrainX)
print(newTrainX.shape)

newTestX=list()
for face_pixels in testX:
	embedding=get_embeddings(model,face_pixels)
	newTestX.append(embedding)

newTestX=asarray(newTestX)
print(newTestX.shape)

savez_compressed('RemontadaEmbeddings.npz', newTrainX, trainy, newTestX, testy)

			

