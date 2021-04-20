'''
Work with the inference engine
'''

import os
import sys
import logging as log 
from openvino.inference_engine import IECore 


class Network:
	'''
	Load and store information for working with the Inference Engine,
	and any loaded models
	'''
	def __init__(self):
		self.plugin=None
		self.network=None
		self.input_blob=None
		self.output_blob=None
		self.exec_network=None
		self.infer_request=None

	def load_model(self,model,device="CPU"):
		'''
		Load the model given IR files
		'''
		model_xml=model
		model_bin=os.path.splitext(model)[0] +'.bin'

		#Initialise the plugin
		self.plugin=IECore()

		#Read the IR as a network
		self.network=self.plugin.read_network(model=model_xml,weights=model_bin)

		#Load the IE Network into the plugin
		self.exec_network=self.plugin.load_network(network=self.network,device_name="CPU")

		#Get the input layer
		self.input_blob=list(self.network.input_info.keys())[0]

		self.output_blob=next(iter(self.network.outputs))

		return

	def get_input_shape(self):
		'''
		Get the input shape
		'''
		return self.network.input_info[self.input_blob].input_data.shape

	def async_inference(self,image):
		'''
		Makes an asynchonours inference request given an input image
		'''
		self.exec_network.start_async(request_id=0,inputs={self.input_blob:image})
		return
	def wait(self):
		'''
		Checks the status of the inference request
		'''
		status=self.exec_network.requests[0].wait(-1)
		return status

	def extract_output(self):
		'''
		Returns a list of the results of the output layer
		'''
		#return self.exec_network.requests[0].outputs['boxes']
		return self.exec_network.requests[0].outputs[self.output_blob]


