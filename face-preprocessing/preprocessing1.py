import cv2
import os
import numpy as np 
from os import listdir
from inference import Network


folder_to_extract="E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/TRAIN/zoe-saldana/"

model="C:/Users/LENOVO/Desktop/Detect&Recognize/intel/face-detection-0202/FP16/face-detection-0202.xml"
#path = "E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/TEST/EXTRACTED-FACES/zoe-saldana/"
path="E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/TRAIN/ex-faces2/zoe-saldana/"


def preprocessing(input_image,height,width):
	preprocessed_image=np.copy(input_image)
	preprocessed_image=cv2.resize(preprocessed_image,(width,height))
	preprocessed_image=preprocessed_image.transpose((2,0,1))
	preprocessed_image=preprocessed_image.reshape(1,3,height,width)
	return preprocessed_image


def extract_faces(image,result,width,height,required_size=(112,112)):

	for box in result[0][0]:
		if box[2]>0.5:
			xmin=int(box[3] *width)
			ymin=int(box[4] *height)
			xmax=int(box[5] *width)
			ymax=int(box[6] * height)

			face=image[ymin:ymax,xmin:xmax]
			face=cv2.resize(face,required_size)

			return face

def model_extraction(model,image):
	
	height,width=image.shape[0],image.shape[1]

	if (height!=None):
	
		plugin=Network()
		plugin.load_model(model=model)
		b,c,h,w=plugin.get_input_shape()
		p_image=preprocessing(image,h,w)
		plugin.async_inference(p_image)
		status=plugin.wait()
		if status==0:
			result=plugin.extract_output()

			face=extract_faces(image,result,width,height)
			return face
	else:
		return np.zeros((160,160,3))

		


def extract_save():
	for filename in listdir(folder_to_extract):
		image_path=folder_to_extract+filename
		image=cv2.imread(image_path)
		face=model_extraction(model=model,image=image)
		cv2.imwrite(path+filename,face)



if __name__=="__main__":
	extract_save()


