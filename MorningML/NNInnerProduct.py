import numpy as np
from NNLayer import NNLayer

class NNInnerProduct(NNLayer):
	def __init__(self, name, nlOut, nlIn):
		NNLayer.__init__(self, name, nlOut, nlIn)


	def init(self):
		print self.name, "No Weights, biases"
		self.W = np.random.randn(self.nlOut, self.nlIn);
		self.B = np.random.randn(self.nlOut, 1);
		self.dw = np.zeros((self.nlOut, self.nlIn))
		self.db = np.zeros((self.nlOut,1));
		print self.W

	def forward(self, inData):
		self.mPrint()
		print self.W.shape
		print self.B.shape
		outData = inData;
		return outData


