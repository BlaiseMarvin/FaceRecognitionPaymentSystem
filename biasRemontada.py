import os 
from os import listdir

from openvino.inference_engine import IECore
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import imutils
import dlib
from numpy import savez_compressed


det_model=r"C:\Users\LENOVO\Desktop\FaceReid\detection_model\face-detection-0202.xml"
det_weights=os.path.splitext(det_model)[0] +'.bin'

#reid_model=r"C:\Users\LENOVO\Desktop\Detect&Recognize\face_net_mobile_face\model-0000.xml"
reid_model=r"E:\FINAL-YEAR-PROJECT\models\ds1\ds-0000.xml"
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


MOT={}


ANC={
	'ben_afflek':"E:/FINAL-YEAR-PROJECT/Bias/archive2/ben_afflek/httpafilesbiographycomimageuploadcfillcssrgbdprgfacehqwMTENDgMDUODczNDcNTcjpg.jpg",
	'elton_john':"E:/FINAL-YEAR-PROJECT/Bias/archive2/elton_john/httpmediapopsugarassetscomfilescbffewltonjpg.jpg",
	'jerry_seinfeld':"E:/FINAL-YEAR-PROJECT/Bias/archive2/jerry_seinfeld/httpimagescontactmusiccomnewsimagesjerryseinfeldjpg.jpg",
	'madonna':"E:/FINAL-YEAR-PROJECT/Bias/archive2/madonna/httpiamediaimdbcomimagesMMVBMTANDQNTAxNDVeQTJeQWpwZBbWUMDIMjQOTYVUXCRALjpg.jpg",
	'mindy_kaling':"E:/FINAL-YEAR-PROJECT/Bias/archive2/mindy_kaling/httpcdncdnjustjaredcomwpcontentuploadsheadlinesmindykalingcomedypilotjpg.jpg"
}

directory="E:/FINAL-YEAR-PROJECT/Bias/archive2"

ben_anchor="E:/FINAL-YEAR-PROJECT/Bias/archive2/ben_afflek/httpafilesbiographycomimageuploadcfillcssrgbdprgfacehqwMTENDgMDUODczNDcNTcjpg.jpg"
elton_anchor="E:/FINAL-YEAR-PROJECT/Bias/archive2/elton_john/httpmediapopsugarassetscomfilescbffewltonjpg.jpg"
jerry_anchor=r"E:\FINAL-YEAR-PROJECT\Bias\archive2\jerry_seinfeld\httpimagescontactmusiccomnewsimagesjerryseinfeldjpg.jpg"
madonna_anchor=r"E:\FINAL-YEAR-PROJECT\Bias\archive2\madonna\httpiamediaimdbcomimagesMMVBMTANDQNTAxNDVeQTJeQWpwZBbWUMDIMjQOTYVUXCRALjpg.jpg"
mindy_anchor=r"E:\FINAL-YEAR-PROJECT\Bias\archive2\mindy_kaling\httpcdncdnjustjaredcomwpcontentuploadsheadlinesmindykalingcomedypilotjpg.jpg"


for name in listdir(directory):
	print('Now on: ',name)
	count=0
	MOT[str(name)]=[]
	path=directory+'/' +name

	

	for file in listdir(path):
		current_location=path+'/' +file
		extracted_ancFace=extract_face(ANC[str(name)])
		test_img=extract_face(current_location)

		emb1=reidentify(extracted_ancFace)
		emb2=reidentify(test_img)

		score=cosine(emb1,emb2)

		MOT[str(name)].append(score)

	print('End of ',name)

		


		

print(MOT)

print("\n")

print("Bias Percentages")
print("\n")

print("At Threshold 0.5")

for name in MOT.keys():
	values=MOT[str(name)]
	masked=[]
	for val in values:
		if val<=0.5:
			masked.append(1)
		else:
			masked.append(0)

	percentage=sum(masked)/len(masked)

	print(name," accuracy: ",percentage)
	print("\n")

print("\n")

print("At Threshold 0.55")

for name in MOT.keys():
	values=MOT[str(name)]
	masked=[]
	for val in values:
		if val<=0.55:
			masked.append(1)
		else:
			masked.append(0)

	percentage=sum(masked)/len(masked)

	print(name," accuracy: ",percentage)
	print("\n")






