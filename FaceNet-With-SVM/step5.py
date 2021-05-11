#Predicting on random images
import matplotlib.pyplot as plt 
import os
import pickle
from inference import Network
import cv2
from os import listdir
import numpy as np 

directory="E:/FINAL-YEAR-PROJECT/models/trainedFacenetModels/pretrainedModel/HallOfFame/"
model="E:/FINAL-YEAR-PROJECT/models/trainedFacenetModels/retrained-413to426-Layers/facenet-413426-LastLayer.xml"

pickle_in=open("model.pickle","rb")
model2=pickle.load(pickle_in)

images=[]
known_labels=['Baguma','Blaise','Fred','Kabwama','Unknown']
new_labels=[]

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



for file in listdir(directory):
	
	img=cv2.imread(directory+file)
	embeddings=get_embeddings(model,img)
	prediction=model2.predict(embeddings)
	new_labels.append(prediction[0])


	img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img=cv2.resize(img,(450,450))
	images.append(img)

plt.figure(figsize=(10,10),dpi=100) # specifying the overall grid size

for i in range(45):
    fig=plt.subplot(9,5,i+1)
    ax=fig.axes

    ax.set_title(new_labels[i])
    ax.title.set_fontsize(10)

    
  

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.imshow(images[i])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()