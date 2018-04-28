import numpy as np
from NNLayer import NNLayer
from NNData import NNData


class NNInnerProduct(NNLayer):
	def __init__(self, network, name, nlOut, nlIn):
		NNLayer.__init__(self, network, name, nlOut, nlIn)


	def initWeights(self):
		print self.name, "No Weights, biases"
		self.W = NNData(self.nlOut, self.nlIn);
		self.W.rand();
		self.B = NNData(self.nlOut, 1);
		self.dw = NNData(self.nlOut, self.nlIn)
		self.db = NNData(self.nlOut,1);
		print self.W.mPrint()
		print self.B.mPrint()

	def forward(self, inData):
		self.mPrint()
		print self.W.data.shape
		print self.B.data.shape
		outData = inData;
		return outData


