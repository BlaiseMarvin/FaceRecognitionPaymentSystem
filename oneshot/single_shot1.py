from openvino.inference_engine import IECore
import cv2
import os
import numpy as np
from scipy.spatial.distance import cosine

import imutils
import dlib
from numpy import savez_compressed










det_model=r"C:\Users\LENOVO\Desktop\FaceReid\detection_model\face-detection-0202.xml"
det_weights=os.path.splitext(det_model)[0] +'.bin'

#reid_model=r"C:\Users\LENOVO\Desktop\FaceNet-BlaiseVersion\20180408-102900.xml"
#reid_weights=os.path.splitext(reid_model)[0] +'.bin'

##MOBILE FACE ARCFACE
##VGGFACE2 FAILED
#reid_model=r"C:\Users\LENOVO\Desktop\Detect&Recognize\face_net_mobile_face\model-0000.xml"
#reid_model=r"C:\Users\LENOVO\Desktop\FaceReid\public\face-recognition-resnet100-arcface\model-r100-ii\model-0000.xml"
#reid_model=r"C:\Users\LENOVO\Desktop\Detect&Recognize\face_net_mobile_face\model-0000.xml"
#reid_model=r"E:\FINAL-YEAR-PROJECT\Models-With_arcface\facenet-arcfaceLastLayer.xml"
#reid_model=r"E:\FINAL-YEAR-PROJECT\Models-With_arcface\facenet-bottleNeck-LastLayer.xml"
#reid_model=r"C:\Users\LENOVO\Desktop\FaceNet-BlaiseVersion\20180408-102900.xml"
#CASIA WEBFACE
#reid_model=r"C:\Users\LENOVO\Desktop\FaceReid\Casia Webface\20180408-102900\20180408-102900\20180408-102900.xml"
reid_model=r"E:\FINAL-YEAR-PROJECT\Bias\models\reidentification_model\face-reidentification-retail-0095.xml"
reid_weights=os.path.splitext(reid_model)[0] +'.bin'








def output_handler(frame,result,height,width):
	faces=list()
	for box in result[0][0]:
		if box[2]>0.5:
			xmin=int(box[3] *width)
			ymin=int(box[4] *height)
			xmax=int(box[5] *width)
			ymax=int(box[6] *height)

			face=frame[ymin:ymax,xmin:xmax]
			face=cv2.resize(face,(128,128))
			faces.append(face)
	return faces


def extract_face(filename):
	plugin=IECore()
	net=plugin.read_network(model=det_model,weights=det_weights)
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


img=extract_face(r"E:\FINAL-YEAR-PROJECT\Bias\code\reidentification_model\extracted_faces\ben_affleck\httpimagesfandangocomrImageRendererredesignstaticimgnoxportraitjpgpcpcpcimagesmasterrepositoryperformerimagespjpg.jpg")
#print(img.shape)

base=r"E:\FINAL-YEAR-PROJECT\Bias\code\reidentification_model\extracted_faces\ben_affleck\httptrwebimgacstanetcxbdddmediasnmediajpg.jpg"
blaise=cv2.imread(r"C:\Users\LENOVO\Desktop\FaceReid\blaise1.jpg")
unknown=cv2.imread(r"C:\Users\LENOVO\Desktop\FaceReid\unknownguy.jpg")

def reidentify(test_subject):
	reid_plugin=IECore()
	reid_net=reid_plugin.read_network(model=reid_model,weights=reid_weights)
	reid_execnet=reid_plugin.load_network(network=reid_net,device_name="CPU")


	reid_inputblob=list(reid_net.input_info.keys())[0]
	reid_outputblob=next(iter(reid_net.outputs))
	b,c,h,w=reid_net.input_info[reid_inputblob].input_data.shape
	p_image=cv2.cvtColor(test_subject,cv2.COLOR_BGR2RGB)
	p_image=cv2.resize(test_subject,(w,h))
	p_image=p_image.transpose((2,0,1))
	p_image=p_image.reshape(1,3,h,w)


	infer_request=reid_execnet.start_async(request_id=0,inputs={reid_inputblob:p_image})
	status=reid_execnet.requests[0].wait(-1)
	if status==0:
		result=reid_execnet.requests[0].outputs[reid_outputblob]

		#This stores embeddings
		#print(result[0])
		#print('storing embedding')
		#savez_compressed('tariq3.npz',result[0])

		#return np.array(result).reshape((1,256))[0]
		return result[0]


def is_match(known_embedding,candidate_embedding,thresh=0.5):
	#calculate the distance between embeddings
	score=cosine(known_embedding,candidate_embedding)
	#score= np.sqrt(np.sum(np.square(np.subtract(known_embedding, candidate_embedding))))
	if score<=thresh:
		print('face is a match',('Score: ',score,' Threshold: ',thresh))
	else:
		print('face is not a match',('Score: ',score,' Threshold: ',thresh))


#z=align_face(r"C:\Users\LENOVO\Desktop\Detect&Recognize\images\train\blaise\IMG_20210126_143516.jpg")
#u=align_face(r"C:\Users\LENOVO\Desktop\FaceNet-BlaiseVersion\facenet1\IMG_20201225_110257.jpg")

#if z is not None and u is not None:

#This is what i commented out
is_match(reidentify(extract_face(base)),reidentify(img))

#is_match(reidentify(unknown),reidentify(blaise))
#print(reidentify(img).shape)


#To generate embeddings
#reidentify(img)
