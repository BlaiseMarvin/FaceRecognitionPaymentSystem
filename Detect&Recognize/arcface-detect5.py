import pickle
import cv2
import os
import numpy as np 
from os import listdir
from numpy import asarray
from openvino.inference_engine import IECore
xmin=0
xmax=0
ymin=0
ymax=0

pickle_in=open("model.pickle","rb")
model2=pickle.load(pickle_in)

detect_model_xml=r"C:\Users\LENOVO\Desktop\Detect&Recognize\intel\face-detection-0202\FP16\face-detection-0202.xml"
detect_model_bin=os.path.splitext(detect_model_xml)[0] +'.bin'

recog_model_xml=r"C:\Users\LENOVO\Desktop\Detect&Recognize\face_net_mobile_face\model-0000.xml"
recog_model_bin=os.path.splitext(recog_model_xml)[0] +'.bin'


def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image
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

def extract_face(frame):
	plugin=IECore()
	net=plugin.read_network(model=detect_model_xml,weights=detect_model_bin)
	exec_net=plugin.load_network(network=net,device_name="CPU")
	input_blob=list(net.input_info.keys())[0]
	output_blob=next(iter(net.outputs))
	b,c,h,w=net.input_info[input_blob].input_data.shape
	

	#image=cv2.imread(filename)
	height=frame.shape[0]
	width=frame.shape[1]

	p_image=cv2.resize(frame,(w,h))
	p_image=p_image.transpose((2,0,1))
	p_image=p_image.reshape(1,3,h,w)


	infer_request=exec_net.start_async(request_id=0,inputs={input_blob:p_image})
	status=exec_net.requests[0].wait(-1)

	if status==0:
		result=exec_net.requests[0].outputs[output_blob]

		return output_handler(frame,result,height,width)


def get_embeddings(face_pixels):
	plugin=IECore()
	net=plugin.read_network(model=recog_model_xml,weights=recog_model_bin)
	exec_net=plugin.load_network(network=net,device_name="CPU")

	input_blob=list(net.input_info.keys())[0]
	output_blob=next(iter(net.outputs))

	b,c,h,w=net.input_info[input_blob].input_data.shape
	preprocessed_image=preprocessing(face_pixels,h,w)

	infer_request=exec_net.start_async(request_id=0,inputs={input_blob:preprocessed_image})
	status=exec_net.requests[0].wait(-1)
	if status==0:
		embz=exec_net.requests[0].outputs[output_blob]
		return embz[0]

#img=cv2.imread("E:/Oh Faces/test2.jpg")
#faces=extract_face(img)

#for face in faces:
#	embeddings=get_embeddings(face).reshape(1,-1)
#	prediction=model2.predict(embeddings)
#	print(prediction)


cap=cv2.VideoCapture(0)

while(cap.isOpened):
	flag,frame=cap.read()

	faces=extract_face(frame)

	for face in faces:
		embeddings=get_embeddings(face).reshape(1,-1)
		prediction=model2.predict(embeddings)

		starty = ymin - 10 if ymin - 10 > 10 else ymin + 10
		frame=cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,0,255),2)
		#cv2.putText(frame,prediction[0],(xmin,starty),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)

	cv2.imshow('frame',frame)
	k=cv2.waitKey(1) & 0xFF
	if k==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()






