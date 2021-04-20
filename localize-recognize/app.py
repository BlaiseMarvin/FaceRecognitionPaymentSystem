import cv2
from inference import Network
import numpy as np 
from scipy.spatial.distance import cosine
from numpy import load 


model=r"C:\Users\LENOVO\Desktop\Detect&Recognize\intel\face-detection-0202\FP16\face-detection-0202.xml"
model_id=r"C:\Users\LENOVO\Desktop\Detect&Recognize\face_net_mobile_face\model-0000.xml"

INPUT_STREAM=r"C:\Users\LENOVO\Downloads\Power Series Finale- Tariq and Ghost Argue.mp4"

def load_embedding():
	embeddings={}
	data=load('omari2.npz')
	data2=load('tariq1.npz')
	embeddings['Omari']=data['arr_0']
	embeddings['Tariq']=data2['arr_0']
	
	
	
	return embeddings


def is_match(known_embedding,candidate_embedding,thresh=0.55):
	#for key in known_embedding.keys():
		#print(key)
	score=cosine(known_embedding['Omari'],candidate_embedding)
	if score<=thresh:
		return 'Omari'
	else:
		score=cosine(known_embedding['Tariq'],candidate_embedding)
		if score<=thresh:
			return 'Tariq'
		else:
			return 'Unknown'
		#pass
		#return 'Unknown'
	
	
	
		

def preprocessing(input_image,height,width):
	preprocessed_image=np.copy(input_image)
	preprocessed_image=cv2.resize(preprocessed_image,(width,height))
	preprocessed_image=preprocessed_image.transpose((2,0,1))
	preprocessed_image=preprocessed_image.reshape(1,3,height,width)
	return preprocessed_image

def perform_facerecognition(face,model):
	plugin=Network()
	plugin.load_model(model=model)
	b,c,h,w=plugin.get_input_shape()
	p_image=preprocessing(face,h,w)
	plugin.async_inference(p_image)
	status=plugin.wait()
	if status==0:
		result=plugin.extract_output()
		candidate_embedding=result[0]
		known_embedding=load_embedding()

		recognized_name=is_match(known_embedding,candidate_embedding)

		return recognized_name





def extract_faces(image,result,width,height):
	
	for box in result[0][0]:
		if box[2]>0.5:
			
			xmin=int(box[3] * width)
			ymin=int(box[4] * height)
			xmax=int(box[5] * width)
			ymax=int(box[6] * height)
			face=image[ymin:ymax,xmin:xmax]
			
			text=perform_facerecognition(face,model_id)

			cv2.putText(image,text,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),2)
			cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255),1)

	return image


def perform_inference():
	

	plugin=Network()
	

	plugin.load_model(model=model)
	

	b,c,h,w=plugin.get_input_shape()
	
	cap=cv2.VideoCapture(INPUT_STREAM)
	while(cap.isOpened()):

		flag,frame=cap.read()
		width=int(cap.get(3))
		height=int(cap.get(4))

		if not flag:
			break


		preprocessed_image=preprocessing(frame,h,w)

		plugin.async_inference(preprocessed_image)

		status=plugin.wait()

		if status==0:
			result=plugin.extract_output()

		#print(result.shape)

			f_image=extract_faces(frame,result,width,height)

			

			cv2.imshow('image',f_image)

		k=cv2.waitKey(1) & 0xFF
		if k==ord('q'):
			break
	cap.release()

	cv2.destroyAllWindows()


def main():
	perform_inference()


if __name__=="__main__":
	main()





