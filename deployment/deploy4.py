from openvino.inference_engine import IECore
import threading
import time
import cv2
import os
from os import listdir
import numpy as np
from inference import Network
import tflite_runtime.interpreter as tflite 
import threading
import multiprocessing
import pickle

'''
Pickle in
'''





'''
Load all TFLite stuff
'''






INPUT_STREAM=r"C:\Users\LENOVO\Downloads\Power Series Finale- Tariq and Ghost Argue.mp4"

#DETECTION MODEL
det_model=r"C:\Users\LENOVO\Desktop\Detect&Recognize\intel\face-detection-0202\FP16\face-detection-0202.xml"

recognizedIdentity=['']

'''
Load up the anchor images
'''


def load_anchors():
	print("Processing Anchors")
	pickle_in=open('theimages.pickle','rb')
	
	return pickle.load(pickle_in)
	
'''
Load the anchors
'''
anchors=load_anchors()



#Preprocessing
def preprocessing(input_image,height,width):
    preprocessed_image=cv2.resize(input_image,(width,height))
    preprocessed_image=preprocessed_image.transpose((2,0,1))
    preprocessed_image=preprocessed_image.reshape(1,3,height,width)
    return preprocessed_image


def perform_facerecognition(face):
	interpreter=tflite.Interpreter('siamese_model.tflite')
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	face=cv2.resize(face,(160,160))
	face=np.expand_dims(face/255.0,axis=0).astype(np.float32)

	
	for (name,data) in anchors.items():
		print(name)
		interpreter.set_tensor(input_details[0]["index"],data)
		interpreter.set_tensor(input_details[1]["index"],face)
		interpreter.invoke()
		predictions=interpreter.get_tensor(output_details[0]["index"])
		predictions=predictions[0][0]
		if predictions>=0.8:
			# print(name)
			recognizedIdentity[0]=name
		# else:
		# 	return 'Unknown'


	# data1=anchors['ghost']

	
	# interpreter.set_tensor(input_details[0]["index"],data1)
	# interpreter.set_tensor(input_details[1]["index"],face)
	# interpreter.invoke()
	# predictions=interpreter.get_tensor(output_details[0]["index"])
	# predictions=predictions[0][0]
	# print(predictions)
	# if predictions>=0.8:
	# 	# recognizedIdentity[0]='tariq'
	# 	return 'Ghost'
	# else:
	# 	return 'Unknown'




	# for (name,data) in anchors.items():
	# 	interpreter.set_tensor(input_details[0]["index"],data)
	# 	interpreter.set_tensor(input_details[1]["index"],face)
	# 	interpreter.invoke()
	# 	predictions=interpreter.get_tensor(output_details[0]["index"])
	# 	# print(name,predictions)
	# 	if predictions[0][0] >=0.9:
	# 		recognizedIdentity[0]=name
	# 		print(recognizedIdentity[0])






#Face detection output
def extract_face(image,result,width,height):
	for box in result[0][0]:
	    if box[2]>0.5:
	        xmin=int(box[3]*width)
	        ymin=int(box[4]*height)
	        xmax=int(box[5]*width)
	        ymax=int(box[6]*height)

	        face=image[ymin:ymax,xmin:xmax]
	        x=threading.Thread(target=perform_facerecognition,args=(face,))
	        x.start()
	        x.join()
	        

			

	        text=recognizedIdentity[0]
	        recognizedIdentity[0]=''	        

	        cv2.putText(image,text,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),2)
	        	        
	        
	        
	        image=cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255),1)
	return image





'''
Detection model settings
'''
plugin=Network()
plugin.load_model(model=det_model)
b,c,h,w=plugin.get_input_shape()



#Open Video Stream
cap=cv2.VideoCapture(INPUT_STREAM)
while(cap.isOpened()):
	flag,frame=cap.read()
	width=int(cap.get(3))
	height=int(cap.get(4))

	if not flag:
		break

	preprocessed_image=preprocessing(frame,h,w)
	plugin.async_inference(preprocessed_image)
	status=plugin.wait()

	if status==0:
		result=plugin.extract_output()
		showThis=extract_face(frame,result,width,height)

		cv2.imshow('frame',showThis)

		k=cv2.waitKey(1) & 0xFF
		if k==ord('q'):
			break

cap.release()
cv2.destroyAllWindows()














