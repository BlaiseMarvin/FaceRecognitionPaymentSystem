import os
from os import listdir
import numpy as np 
from numpy import asarray
import cv2
from openvino.inference_engine import IECore
from numpy import savez_compressed

model_xml=r"C:\Users\LENOVO\Desktop\Detect&Recognize\intel\face-detection-0202\FP16\face-detection-0202.xml"
model_bin=os.path.splitext(model_xml)[0] +'.bin'


def output_handler(frame,result,height,width):
	faces=list()
	for box in result[0][0]:
		if box[2] >=0.5:
			xmin=int(box[3] * width)
			ymin=int(box[4] *height)
			xmax=int(box[5] * width)
			ymax=int(box[6] *height)

			face=frame[ymin:ymax,xmin:xmax]
			face=cv2.resize(face,(112,112))

			faces.append(face)

	return faces




def extract_face(filename):
	plugin=IECore()
	net=plugin.read_network(model=model_xml,weights=model_bin)
	exec_net=plugin.load_network(network=net,device_name="CPU")
	input_blob=list(net.input_info.keys())[0]
	output_blob=next(iter(net.outputs))
	b,c,h,w=net.input_info[input_blob].input_data.shape
	

	image=cv2.imread(filename)
	height=image.shape[0]
	width=image.shape[1]

	p_image=cv2.resize(image,(w,h))
	p_image=p_image.transpose((2,0,1))
	p_image=p_image.reshape(1,3,h,w)


	infer_request=exec_net.start_async(request_id=0,inputs={input_blob:p_image})
	status=exec_net.requests[0].wait(-1)

	if status==0:
		result=exec_net.requests[0].outputs[output_blob]

		return output_handler(image,result,height,width)[0]



def load_faces(directory):
	faces=list()

	#enumerate files
	for filename in listdir(directory):
		#path
		path=directory+filename
		#get face
		face=extract_face(path)
		#store
		faces.append(face)
	return faces


def load_dataset(directory):
	X,y=list(),list()

	#enumerate folders on per class
	for subdir in listdir(directory):
		#path
		path=directory + subdir + '/'

		#load all faces in the subdirectory
		faces=load_faces(path)

		#create labels
		labels=[subdir for _ in range(len(faces))]

		#Summarize progress
		print('> loaded %d examples for class: %s' %(len(faces),subdir))

		#store
		X.extend(faces)
		y.extend(labels)

	return asarray(X),asarray(y)

trainX,trainy=load_dataset("C:/Users/LENOVO/Desktop/Detect&Recognize/images/train/")
print(trainX.shape, trainy.shape)

testX,testy=load_dataset("C:/Users/LENOVO/Desktop/Detect&Recognize/images/val/")
savez_compressed('blaise-unknown-dataset.npz', trainX, trainy, testX, testy)



