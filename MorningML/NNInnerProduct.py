import numpy as np
from NNLayer import NNLayer
from NNData import NNData


class NNInnerProduct(NNLayer):
	def __init__(self, network, name, nlOut):
		NNLayer.__init__(self, network, name, nlOut)


	def initWeights(self):
		print self.name, "No Weights, biases"
		self.W = NNData(self.nlOut, self.nlIn);
		#self.W.rand();
		self.W.ones();
		self.B = NNData(self.nlOut, 1);
		self.dw = NNData(self.nlOut, self.nlIn)
		self.db = NNData(self.nlOut,1);


	def forward(self, inData):
		self.outData = self.W * inData + self.B;
		return self.outData


