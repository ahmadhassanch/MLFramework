import numpy as np
from NNLayer import NNLayer

class NNInnerProduct(NNLayer):
	def __init__(self, name, nlOut, nlIn):
		NNLayer.__init__(self, name, nlOut, nlIn)
		self.weights = np.zeros((nlOut, nlIn))
		self.bias = np.zeros((nlOut,1));
		self.dw = np.zeros((nlOut, nlIn))
		self.db = np.zeros((nlOut,1));


	def forward(self):
		self.mPrint()
		print self.weights.shape
		print self.bias.shape
