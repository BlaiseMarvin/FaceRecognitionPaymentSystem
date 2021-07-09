
#INPUT_STREAM (Since I'm not testing on live video yet)
# INPUT_STREAM=r"C:\Users\LENOVO\Downloads\Power Series Finale- Tariq and Ghost Argue.mp4"
# INPUT_STREAM=r"C:\Users\LENOVO\Downloads\Video\videoplayback.mp4"


#Necessary Imports
from openvino.inference_engine import IECore
import cv2
import os
import numpy as np
import pickle
import threading
from multiprocessing import Process
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
import time


#Detection model architecture
det_model=r"C:\Users\LENOVO\Desktop\Detect&Recognize\intel\face-detection-0202\FP16\face-detection-0202.xml"
det_weights=os.path.splitext(det_model)[0]+'.bin'

#Siamese-recognition model
# recogModel=r"C:\Users\LENOVO\Downloads\siamese_model\siamese_model.xml"
recogModel=r"E:\FINAL-YEAR-PROJECT\siamese_networks\intel-savedModel\faceNetOneshotJune.xml"
recogWeights=os.path.splitext(recogModel)[0]+'.bin'


#Instantiate the plugin
plugin=IECore()


'''
Prepare the detection model
'''
detPlugin=plugin
detNet=detPlugin.read_network(model=det_model,weights=det_weights)
detExecNet=detPlugin.load_network(network=detNet,device_name="MYRIAD")
det_input_blob=list(detNet.input_info.keys())[0]
det_output_blob=next(iter(detNet.outputs))
db,dc,dh,dw=detNet.input_info[det_input_blob].input_data.shape

'''
Prepare the recognition model
'''
recogPlugin=plugin
recogNetwork=recogPlugin.read_network(model=recogModel,weights=recogWeights)
recogExecNet=recogPlugin.load_network(network=recogNetwork,device_name="MYRIAD")
recog_input_blob1=list(recogNetwork.input_info.keys())[0]
recog_input_blob2=list(recogNetwork.input_info.keys())[1]
recog_output_blob=next(iter(recogNetwork.outputs))
b1,c1,h1,w1=recogNetwork.input_info[recog_input_blob1].input_data.shape

#Load up all anchor images-
def load_anchors():
	pickle_in=open('anchors.pickle','rb')
	return pickle.load(pickle_in)


#Preprocessing: Preprocess the frame for the model
def preprocessing(input_image,height,width):
	try:
	    preprocessed_image=cv2.resize(input_image,(width,height))
	    preprocessed_image=preprocessed_image.transpose((2,0,1))
	    preprocessed_image=preprocessed_image.reshape(1,3,height,width)
	    return preprocessed_image
	except:
		pass

#Deduct the bus fare from the walletBalance
def deduct_fare(id):
	db.collection('facePay').document(id).update({'walletBalance':firestore.Increment(-fare)})


def perform_facerecognition(face):

	#Preprocess face to match model requirements
	p_face=preprocessing(face/255.0,h1,w1)

	for name,values in anchors.items():

		p_image=preprocessing(values['face']/255.0,h1,w1)


		infer_req=recogExecNet.start_async(request_id=0,inputs={recog_input_blob1:p_face,recog_input_blob2:p_image})
		status=recogExecNet.requests[0].wait(-1)

		if status==0:

			if recogExecNet.requests[0].outputs[recog_output_blob][0][0]>=0.85:

				recognizedIdentity[0]=name
				if values['walletBalance']>=fare:

					positiveTransaction[0]=''.join(['Success',' ',name])
					if anchors[name]['state']!=1:
						y=threading.Thread(target=deduct_fare,args=(values['id'],))
						y.start()
					anchors[name]['state']=1

				else:
					negativeTransaction[0]=''.join([name,' ','your balance is Insufficient'])





					





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
			poztxt=positiveTransaction[0]
			positiveTransaction[0]=''
			negtxt=negativeTransaction[0]
			negativeTransaction[0]=''
			recognizedIdentity[0]=''
			cv2.putText(image,text,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),2)


			'''
			Put additional text to the screen
			'''

			cv2.putText(image, 
                poztxt, 
                (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                (0,255,255), 
                2, 
                cv2.LINE_4)
			cv2.putText(image, 
                negtxt, 
                (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                (0,255,255), 
                2, 
                cv2.LINE_4)

			image=cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255),1)

	return image

#On snapshot callback
def on_snapshot(doc_snapshot,changes,read_time):
	for doc in doc_snapshot:
		anchors[doc.to_dict()['userName']]['walletBalance']=doc.to_dict()['walletBalance']
	callback_done.set()




if __name__=="__main__":

	#Firestore stuff
	cred=credentials.Certificate("serviceAccountKey.json")
	firebase_admin.initialize_app(cred)
	db=firestore.client()

	#Listen to live changes
	callback_done=threading.Event()

	#Listen to only documents with activated FacePay
	col_query=db.collection('facePay').where('activatedFacePay','==',True)
	query_watch=col_query.on_snapshot(on_snapshot)

	#fare
	fare=30000
	#recognized_identity
	recognizedIdentity=['']
	#positiveTransaction
	positiveTransaction=['']
	#Negative transaction
	negativeTransaction=['']
	#Load all anchors
	anchors=load_anchors()
	#Video Inference: 
	# cap=cv2.VideoCapture(INPUT_STREAM)
	cap=cv2.VideoCapture(0)

	while(cap.isOpened()):
		flag,frame=cap.read()
		if not flag:
			break
		width=int(cap.get(3))
		height=int(cap.get(4))

		pimage=preprocessing(frame,dh,dw)
		det_infer_request=detExecNet.start_async(request_id=0,inputs={det_input_blob:pimage})
		status=detExecNet.requests[0].wait(-1)

		if status==0:
			result=detExecNet.requests[0].outputs[det_output_blob]
			image=extract_face(frame,result,width,height)

			cv2.imshow('frame', image)

			k=cv2.waitKey(1) & 0xFF
			if k==ord('q'):
				pickle_out=open('anchors.pickle','wb')
				pickle.dump(anchors,pickle_out)
				pickle_out.close
				break
	cap.release()
	cv2.destroyAllWindows()






