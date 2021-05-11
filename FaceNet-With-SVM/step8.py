import numpy as np
import cv2
import pickle
from inference import Network

known_labels=['Baguma','Blaise','Fred','Kabwama','Unknown']
detection_model=r"C:\Users\LENOVO\Desktop\Detect&Recognize\intel\face-detection-0202\FP16\face-detection-0202.xml"
model_id="E:/FINAL-YEAR-PROJECT/models/trainedFacenetModels/retrained-413to426-Layers/facenet-413426-LastLayer.xml"
INPUT_STREAM="E:/FINAL-YEAR-PROJECT/models/trainedFacenetModels/pretrainedModel/VID_20210427_130039.mp4"

pickle_in=open("model.pickle","rb")
model2=pickle.load(pickle_in)

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
		return candidate_embedding
		

		

def extract_faces(image,result,width,height):
	
	for box in result[0][0]:
		if box[2]>0.5:
			
			xmin=int(box[3] * width)
			ymin=int(box[4] * height)
			xmax=int(box[5] * width)
			ymax=int(box[6] * height)
			face=image[ymin:ymax,xmin:xmax]
			
			embz=perform_facerecognition(face,model_id)
			prediction=model2.predict(embz.reshape(1,-1))

			label=known_labels[prediction[0]]


			cv2.putText(image,label,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
			cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255),1)

	return image

def perform_inference():
	

	plugin=Network()
	

	plugin.load_model(model=detection_model)
	

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



