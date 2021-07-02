from openvino.inference_engine import IECore
import threading
import time
from multiprocessing import Process 
import pickle
import cv2
import os
import numpy as np
from scipy.spatial.distance import cosine
from numpy import load 



recognizedIdentity=['']

INPUT_STREAM=r"C:\Users\LENOVO\Downloads\Power Series Finale- Tariq and Ghost Argue.mp4"

#detection model
det_model=r"C:\Users\LENOVO\Desktop\Detect&Recognize\intel\face-detection-0202\FP16\face-detection-0202.xml"
det_weights=os.path.splitext(det_model)[0]+'.bin'

#recognition model
recogModel=r"C:\Users\LENOVO\Desktop\Detect&Recognize\face_net_mobile_face\model-0000.xml"
recogweights=os.path.splitext(recogModel)[0]+'.bin'

#Load the plugin
plugin=IECore()



'''
Preparing the recognition model for the inference engine
'''

recogPlugin=plugin
recogNet=recogPlugin.read_network(model=recogModel,weights=recogweights)
recogExecNet=recogPlugin.load_network(network=recogNet,device_name="MYRIAD")
recog_input_blob=list(recogNet.input_info.keys())[0]
recog_output_blob=next(iter(recogNet.outputs))
rb,rc,rh,rw=recogNet.input_info[recog_input_blob].input_data.shape


'''
Prepraring the detection model for the inference engine
'''

detPlugin=plugin
detNet=detPlugin.read_network(model=det_model,weights=det_weights)
detExecNet=detPlugin.load_network(network=detNet,device_name="MYRIAD")
det_input_blob=list(detNet.input_info.keys())[0]
det_output_blob=next(iter(detNet.outputs))
db,dc,dh,dw=detNet.input_info[det_input_blob].input_data.shape


def load_embedding():
    pickle_in=open('userEmbeddings.pickle','rb')

    return pickle.load(pickle_in)

def is_match(known_embedding,candidate_embedding,thresh=0.55):
    
    for(name,embedding) in known_embedding.items():
        
        score=cosine(embedding,candidate_embedding)
        
        if score<=thresh:
            print(name)
            
            recognizedIdentity[0]=name
           
        # else:
            
        #     recognizedIdentity.append('Unknown')

        # print(recognizedIdentity)
     


def preprocessing(input_image,height,width):
    preprocessed_image=cv2.resize(input_image,(width,height))
    preprocessed_image=preprocessed_image.transpose((2,0,1))
    preprocessed_image=preprocessed_image.reshape(1,3,height,width)
    return preprocessed_image


def perform_facerecognition(face):
    p_image=preprocessing(face,rh,rw)
    recog_infer_request=recogExecNet.start_async(request_id=0,inputs={recog_input_blob:p_image})
    status=recogExecNet.requests[0].wait(-1)

    
    if status==0:
        result=recogExecNet.requests[0].outputs[recog_output_blob]
        candidate_embedding=result[0]
        known_embedding=load_embedding()

        x=threading.Thread(target=is_match,daemon=True,args=(known_embedding,candidate_embedding,))
        x.start()
        x.join()
        
        return recognizedIdentity[0]
        

def extract_face(image,result,width,height):
    for box in result[0][0]:
        if box[2]>0.5:
            xmin=int(box[3]*width)
            ymin=int(box[4]*height)
            xmax=int(box[5]*width)
            ymax=int(box[6]*height)

            face=image[ymin:ymax,xmin:xmax]
            text=perform_facerecognition(face)
            recognizedIdentity[0]=''

            cv2.putText(image,text,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),2)
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255),1)
            
            
            
            image=cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255),1)
    return image
        

cap=cv2.VideoCapture(INPUT_STREAM)

while(cap.isOpened()):
    flag,frame=cap.read()
    width=int(cap.get(3))
    height=int(cap.get(4))


    pimage=preprocessing(frame,dh,dw)
    det_infer_request=detExecNet.start_async(request_id=0,inputs={det_input_blob:pimage})
    status=detExecNet.requests[0].wait(-1)

    if status==0:
        result=detExecNet.requests[0].outputs[det_output_blob]
        img=extract_face(frame,result,width,height)
    
        cv2.imshow('frame',img)
    
        k=cv2.waitKey(1) & 0xFF
        if k==ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()

