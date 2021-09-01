from fastapi import FastAPI
import cv2
import base64
import io
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from pydantic import BaseModel


#Initialize the MTCNN face detector
detector=MTCNN()

app=FastAPI()

class detectFace(BaseModel):
   image:str

class ExtractFace(BaseModel):
   image:str

#Base64 string to image
def base64str_to_PILImage(base64str):
   base64_img_bytes = base64str.encode('utf-8')
   base64bytes = base64.b64decode(base64_img_bytes)
   bytesObj = io.BytesIO(base64bytes)
   img = Image.open(bytesObj)
   return img

#Home
@app.get('/')
def home():
   return {'data':{'message':'KPES facePay'}}

#Detect Face In Image
@app.post('/detectFace')
def detectFace(request:detectFace):
    
   img=asarray(base64str_to_PILImage(request.image))
   faces=detector.detect_faces(img)
   if len(faces)==1:
      return {'data':{'message':'Success','box':faces[0]['box'],'confidence':faces[0]['confidence']}}
   else:
      return {'data':{'message':'Error, more than one face detected'}}


#Extract Face and store embeddings in db
#Takes a single image, extracts face and stores facestring in database
@app.post('/extractFace')
def extractFace(request:ExtractFace):
   try:
  
      img=asarray(base64str_to_PILImage(request.image))

      faces=detector.detect_faces(img)

      x,y,width,height=faces[0]['box']
      x2=x+width
      y2=y+height
      face=img[y:y2,x:x2]
      face=cv2.resize(face,(160,160))
      retval,buffer=cv2.imencode('.jpg',face)
      facestring=base64.b64encode(buffer).decode('utf-8')

      return {'data':{'message':'Success','face':facestring}}
   except:
      return {'data':{'message':'Error, please try again'}}
   
      


#Perform Face Recognition Using a Siamese Tflite model


   






   





   
   
    
