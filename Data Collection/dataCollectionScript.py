#Detect and extract webcam faces
import cv2
import os
import numpy as np 
from os import listdir
from inference import Network

model="C:/Users/LENOVO/Desktop/Detect&Recognize/intel/face-detection-0202/FP16/face-detection-0202.xml"

#Input frames to the model should be preprocessed

def preprocessing(input_image,height,width):
	preprocessed_image=np.copy(input_image)
	preprocessed_image=cv2.resize(preprocessed_image,(width,height))
	preprocessed_image=preprocessed_image.transpose((2,0,1))
	preprocessed_image=preprocessed_image.reshape(1,3,height,width)
	return preprocessed_image

#Preprocessing the model's output
def output_handler(image,result,width,height,required_size=(160,160)):

	for box in result[0][0]:
		if box[2]>0.5:
			xmin=int(box[3] *width)
			ymin=int(box[4] *height)
			xmax=int(box[5] *width)
			ymax=int(box[6] * height)

			image=cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255),2)

			cropped_face=image[ymin:ymax,xmin:xmax]
			face=cv2.resize(cropped_face,required_size)
			return face


#Leveraging the face detection model from OpenVino
def detect_this(model,image,height,width):
	
	plugin=Network()
	plugin.load_model(model=model)
	b,c,h,w=plugin.get_input_shape()
	p_image=preprocessing(image,h,w)
	plugin.async_inference(p_image)
	status=plugin.wait()
	if status==0:
		result=plugin.extract_output()
		face=output_handler(image,result,width,height)
		return face
		






def main():
	cap=cv2.VideoCapture(0)
	count=0

	#Collecting a specified number of samples
	while(cap.isOpened()):
		count+=1
		flag,frame=cap.read()

		width=int(cap.get(3))
		height=int(cap.get(4))

		if not flag:
			break

		face=detect_this(model,frame,height,width)

		#Storing face in particular folder
		file_name_path='./Images/' + str(count) +'.jpg'
		cv2.imwrite(file_name_path,face)

		#Put a count on images and display live count
		cv2.putText(frame,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
		cv2.imshow('frame',frame)
		k=cv2.waitKey(1) & 0xFF
		if k==ord('q'):
			break


if __name__=="__main__":
	main()



