import cv2
import numpy as np 
from openvino.inference_engine import IECore
import os

model=r"C:\Users\LENOVO\Desktop\Detect&Recognize\intel\face-detection-0202\FP16\face-detection-0202.xml"




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

def draw_boxes(frame,result,width,height):
	fiace=list()

	for box in result[0][0]:
		conf=box[2]
		if conf>=0.5:
			xmin=int(box[3] * width)
			ymin=int(box[4] *height)
			xmax=int(box[5] *width)
			ymax=int(box[6] *height)

			fiace.append(frame[ymin:xmin,ymax:xmax])
			
			cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,0,255),5)
	return frame





plugin= IECore()
model_bin=os.path.splitext(model)[0] +'.bin'
net=plugin.read_network(model=model,weights=model_bin)
exec_net=plugin.load_network(network=net,device_name="CPU")
input_blob=list(net.input_info.keys())[0]
output_blob=next(iter(net.outputs))


b,c,h,w=net.input_info[input_blob].input_data.shape


image=cv2.imread('download.jpg')
height=image.shape[0]
width=image.shape[1]

p_frame=cv2.resize(image,(w,h))
p_frame=p_frame.transpose((2,0,1))
p_frame=p_frame.reshape(1, *p_frame.shape)





#plugin.async_inference(p_frame)
infer_request=exec_net.start_async(request_id=0,inputs={input_blob:p_frame})
status=exec_net.requests[0].wait(-1)

if status==0:
	result=exec_net.requests[0].outputs[output_blob]


	frame=draw_boxes(image,result,width,height)
	frame=cv2.resize(frame,(300,300))
	

	cv2.imshow('frame',frame)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	




