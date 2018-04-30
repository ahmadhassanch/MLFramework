import numpy as np
from NNLayer import NNLayer
from NNData import NNData

class NNInnerProduct(NNLayer):
	def __init__(self, network, name, nlOut):
		NNLayer.__init__(self, network, name, nlOut)

	def initWeights(self):
		print self.name, "Initializing Weights, biases"
		self.W = NNData(self.nlOut, self.nlIn);
		self.B = NNData(self.nlOut, 1);
		self.W.rand();
		#self.W.ones();
		#self.B.data[0,0]=10;
		#print self.W.data.shape;
		self.W.mPrint();
		self.B.mPrint();

	def forward(self, X):		
		self.outData = self.W * X + self.B;
		return self.outData

	def backprop(self, dGlobal):
		m = dGlobal.shape[1];
		W = self.W.data;
		dGlobalNew = 1.0/m * np.matmul(W.T, dGlobal);
		#elf.dw = NNData(self.nlOut, self.nlIn)
		#elf.db = NNData(self.nlOut,1);
	
		return dGlobalNew
