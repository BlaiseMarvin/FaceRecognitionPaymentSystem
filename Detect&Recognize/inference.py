import os
import cv2
from openvino.inference_engine import IECore

def output_handler(frame,result,height,width):
	faces=list()
	for box in result[0][0]:
		if box[2] >=0.5:
			xmin=int(box[3] * width)
			ymin=int(box[4] *height)
			xmax=int(box[5] * width)
			ymax=int(box[6] *height)

			face=frame[ymin:ymax,xmin:xmax]

			faces.append(face)

	return faces



model_xml=r"C:\Users\LENOVO\Desktop\Detect&Recognize\intel\face-detection-0202\FP16\face-detection-0202.xml"
model_bin=os.path.splitext(model_xml)[0] +'.bin'

plugin=IECore()
net=plugin.read_network(model=model_xml,weights=model_bin)
exec_net=plugin.load_network(network=net,device_name="CPU")

input_blob=list(net.input_info.keys())[0]
output_blob=next(iter(net.outputs))

b,c,h,w=net.input_info[input_blob].input_data.shape

image=cv2.imread(r"C:\Users\LENOVO\Desktop\FaceNet-BlaiseVersion\facenet1\download (1).jpg")
height=image.shape[0]
width=image.shape[1]



p_image=cv2.resize(image,(w,h))
p_image=p_image.transpose((2,0,1))
p_image=p_image.reshape(1,3,h,w)


infer_request=exec_net.start_async(request_id=0,inputs={input_blob:p_image})

status=exec_net.requests[0].wait(-1)

if status==0:
	result=exec_net.requests[0].outputs[output_blob]
	for fiace in output_handler(image,result,height,width):
		fiace=cv2.resize(fiace,(300,300))
		cv2.imshow('face',fiace)
		cv2.waitKey(0)
	cv2.destroyAllWindows()



