import cv2
import pickle
from inference import Network
import numpy as np

pickle_in=open("model.pickle","rb")
model2=pickle.load(pickle_in)

model="E:/FINAL-YEAR-PROJECT/models/trainedFacenetModels/pretrainedModel/optimized-reverse-input/facenet-Original-LastLayer.xml"
path=r"E:\FINAL-YEAR-PROJECT\models\decodeThis\0.jpg"


def preprocessing(input_image,height,width):
	preprocessed_image=np.copy(input_image)
	preprocessed_image=cv2.resize(preprocessed_image,(width,height))
	preprocessed_image=preprocessed_image.transpose((2,0,1))
	preprocessed_image=preprocessed_image.reshape(1,3,height,width)
	return preprocessed_image

def get_embeddings(model,face_pixels):
	face_pixels=face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std

	plugin=Network()
	plugin.load_model(model=model)
	b,c,h,w=plugin.get_input_shape()
	preprocessed_image=preprocessing(face_pixels,h,w)
	plugin.async_inference(preprocessed_image)
	status=plugin.wait()
	if status==0:
		embz=plugin.extract_output()
		return embz[0].reshape(1,-1)



img=cv2.imread(path)
embedding=get_embeddings(model,img)

prediction=model2.predict(embedding)
print(prediction)



