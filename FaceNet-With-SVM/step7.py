import numpy as np
import cv2
import pickle
from inference import Network


cap = cv2.VideoCapture("E:/FINAL-YEAR-PROJECT/models/trainedFacenetModels/pretrainedModel/VID_20210427_130039.mp4")
caffeModel = "C:/Users/LENOVO/Desktop/KMC Internship/releaseTheKraken/res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "C:/Users/LENOVO/Desktop/KMC Internship/releaseTheKraken/deploy.prototxt.txt"

known_labels=['Baguma','Blaise','Fred','Kabwama','Unknown']

model=r"C:\Users\LENOVO\Desktop\Detect&Recognize\face_net_mobile_face\model-0000.xml"





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
        return embz[0]



net = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)

while (True):
    pickle_in=open("model.pickle","rb")
    model2=pickle.load(pickle_in)
    flag, frame = cap.read()
    if flag is not True:
        break
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        #print(confidence)

        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cropped_face=frame[startY:endY,startX:endX]
        embeddings=get_embeddings(model,cropped_face)
        prediction=model2.predict(embeddings.reshape(1,-1))
        label=str(prediction[0])
        


        cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if (key == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
