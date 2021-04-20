import os
import cv2
from openvino.inference_engine import IECore

model_xml=r"C:\Users\LENOVO\Desktop\Detect&Recognize\face_net_mobile_face\model-0000.xml"
model_bin=os.path.splitext(model_xml)[0] +'.bin'

plugin=IECore()
net=plugin.read_network(model=model_xml,weights=model_bin)
input_blob=list(net.input_info.keys())[0]
output_blob=next(iter(net.outputs))

exec_net=plugin.load_network(network=net,device_name="CPU")

input_shape=net.input_info[input_blob].input_data.shape 
print(input_shape)

out=exec_net.requests[0].outputs[output_blob]
print(out[0])