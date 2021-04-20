import pickle
import cv2
import os
import numpy as np 
from os import listdir
from numpy import asarray
from openvino.inference_engine import IECore

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
	boundaries=list()
	for box in result[0][0]:
		if box[2] >=0.5:
			xmin=int(box[3] * width)
			boundaries.append(xmin)
			ymin=int(box[4] *height)
			boundaries.append(ymin)
			xmax=int(box[5] * width)
			boundaries.append(xmax)
			ymax=int(box[6] *height)
			boundaries.append(ymax)

			face=frame[ymin:ymax,xmin:xmax]
			face=cv2.resize(face,(112,112))

			faces.append(face)

	return faces,boundaries

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

	#p_image=cv2.resize(frame,(w,h))
	#p_image=p_image.transpose((2,0,1))
	#p_image=p_image.reshape(1,3,h,w)
	p_image=preprocessing(frame,h,w)


	infer_request=exec_net.start_async(request_id=0,inputs={input_blob:p_image})
	status=exec_net.requests[0].wait(-1)

	if status==0:
		result=exec_net.requests[0].outputs[output_blob]
		extracted_face,face_boundaries=output_handler(frame,result,height,width)

		#return output_handler(frame,result,height,width)
		return extracted_face,face_boundaries


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

img=cv2.imread("E:/Oh Faces/test2.jpg")
faces,faces_boundaries=extract_face(img)
print(faces_boundaries[0:4][0])
i=0
y=4
for face in faces:
	#x,y,w,h=cv2.boundingRect(face)
	#x2,y2=x+w,y+h
	stuff=faces_boundaries[i:y]
	if len(stuff)==4:
		x,y,x2,y2=stuff[0],stuff[1],stuff[2],stuff[3]
		i=i+5
		y=y+5

	starty=y-10 if y-10>10 else y +10
	embeddings=get_embeddings(face).reshape(1,-1)
	prediction=model2.predict(embeddings)
	#print(prediction)
	cv2.rectangle(img,(x,y),(x2,y2),(0,0,255),2)
	#cv2.putText(img,prediction[0],(x,starty),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#cap=cv2.VideoCapture(0)

#while(cap.isOpened):
#	flag,frame=cap.read()

#	faces=extract_face(frame)

#	for face in faces:



