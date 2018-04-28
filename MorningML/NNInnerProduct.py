import numpy as np
from NNLayer import NNLayer

class NNInnerProduct(NNLayer):
	def __init__(self, name, nlOut, nlIn):
		NNLayer.__init__(self, name, nlOut, nlIn)

		self.dw = np.zeros((nlOut, nlIn))
		self.db = np.zeros((nlOut,1));

	def init(self):
		print self.name, "No Weights, biases".py
		self.weights = np.random.randn(self.nlOut, self.nlIn);
		self.bias    = np.random.randn(self.nlOut, 1);
		print self.weights

	def forward(self, inData):
		self.mPrint()
		print self.weights.shape
		print self.bias.shape
		outData = inData;
		return outData


