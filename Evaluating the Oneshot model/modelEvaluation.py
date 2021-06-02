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

reid_model=r"C:\Users\LENOVO\Desktop\Detect&Recognize\face_net_mobile_face\model-0000.xml"
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
			face=cv2.resize(face,(112,112))
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
	#'anthony-mackie':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/anthony-mackie/Anthony Mackie132_459.jpg",
	#'daniel-kaluuya':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/daniel-kaluuya/Daniel-Kaluuya (3).jpg",
	#'idris-elba':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/idris-elba/idris-elba-the-gospel-2005-BPPBJE.jpg",
	#'kanye-west':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/kanye-west/Kanye-West (1).jpg",
	#'lupita':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/lupita/beauty-2015-02-oscars-2015-beauty-lupita-nyongo-main.jpg"
	#'michael-blackson':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/michael-blackson/503862_v9_bb.jpg",
	#'morgan-freeman':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/morgan-freeman/2402.jpg",
	#'obama':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/obama/barack obama40_712.jpg",
	#'olivia-pope':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/olivia-pope/download (1).jpg"
	'rihanna':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/rihanna/images (1).jpg"
	#'thiery-henry':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/thiery-henry/images (62).jpg",
	#'viola-davis':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/viola-davis/‘HTGAWM’s-Viola-Davis-Why-Playing-Annalise-Keating-Has-‘Meant-Everything.jpg",
	#'will-smith':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/will-smith/download (3).jpg",
	#'zendaya':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/zendaya/591658_v9_bb.jpg",
	#'zoe-saldana':"E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/zoe-saldana/e708c468969d68c966422f5962e7f69453-2-zoe-saldana.2x.rhorizontal.w710.jpg"


	}

directory="E:/FINAL-YEAR-PROJECT/Dataset++/face-dataset/evaluateThis/TakeFour"




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

	percentage=sum(masked)/(len(masked)-1)

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

	percentage=sum(masked)/(len(masked)-1)

	print(name," accuracy: ",percentage)
	print("\n")

print("\n")
print("At Threshold 0.6")

for name in MOT.keys():
	values=MOT[str(name)]
	masked=[]
	for val in values:
		if val<=0.6:
			masked.append(1)
		else:
			masked.append(0)

	percentage=sum(masked)/(len(masked)-1)

	print(name," accuracy: ",percentage)
	print("\n")

print("\n")
print("At Threshold 0.65")

for name in MOT.keys():
	values=MOT[str(name)]
	masked=[]
	for val in values:
		if val<=0.65:
			masked.append(1)
		else:
			masked.append(0)

	percentage=sum(masked)/(len(masked)-1)

	print(name," accuracy: ",percentage)
	print("\n")

print("\n")
print("At Threshold 0.70")

for name in MOT.keys():
	values=MOT[str(name)]
	masked=[]
	for val in values:
		if val<=0.70:
			masked.append(1)
		else:
			masked.append(0)

	percentage=sum(masked)/(len(masked)-1)

	print(name," accuracy: ",percentage)
	print("\n")

print("\n")
print("At Threshold 0.75")

for name in MOT.keys():
	values=MOT[str(name)]
	masked=[]
	for val in values:
		if val<=0.75:
			masked.append(1)
		else:
			masked.append(0)

	percentage=sum(masked)/(len(masked)-1)

	print(name," accuracy: ",percentage)
	print("\n")

print("\n")
print("At Threshold 0.80")

for name in MOT.keys():
	values=MOT[str(name)]
	masked=[]
	for val in values:
		if val<=0.80:
			masked.append(1)
		else:
			masked.append(0)

	percentage=sum(masked)/(len(masked)-1)

	print(name," accuracy: ",percentage)
	print("\n")

print("\n")
print("At Threshold 0.85")

for name in MOT.keys():
	values=MOT[str(name)]
	masked=[]
	for val in values:
		if val<=0.85:
			masked.append(1)
		else:
			masked.append(0)

	percentage=sum(masked)/(len(masked)-1)

	print(name," accuracy: ",percentage)
	print("\n")

print("\n")
print("At Threshold 0.90")

for name in MOT.keys():
	values=MOT[str(name)]
	masked=[]
	for val in values:
		if val<=0.90:
			masked.append(1)
		else:
			masked.append(0)

	percentage=sum(masked)/(len(masked)-1)

	print(name," accuracy: ",percentage)
	print("\n")







