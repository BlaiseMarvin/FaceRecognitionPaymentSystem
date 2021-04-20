import os
import cv2
import numpy as np 
from numpy import asarray
from openvino.inference_engine import IECore
from numpy import load 
from numpy import savez_compressed

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

model_xml=r"C:\Users\LENOVO\Desktop\Detect&Recognize\face_net_mobile_face\model-0000.xml"
model_bin=os.path.splitext(model_xml)[0] +'.bin'

def get_embeddings(face_pixels):
	plugin=IECore()
	net=plugin.read_network(model=model_xml,weights=model_bin)
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


data=load('blaise-unknown-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

newTrainX=list()

for face_pixels in trainX:
	face_pixels=cv2.cvtColor(face_pixels,cv2.COLOR_BGR2RGB)
	embedding=get_embeddings(face_pixels)
	newTrainX.append(embedding)

newTrainX=asarray(newTrainX)
print(newTrainX.shape)

newTestX=list()
for face_pixels in testX:
	face_pixels=cv2.cvtColor(face_pixels,cv2.COLOR_BGR2RGB)
	embedding=get_embeddings(face_pixels)
	newTestX.append(embedding)

newTestX=asarray(newTestX)
print(newTestX.shape)

savez_compressed('blaise-unknown-embeddings.npz', newTrainX, trainy, newTestX, testy)




