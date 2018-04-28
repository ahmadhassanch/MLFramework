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

		#self.dw = NNData(self.nlOut, self.nlIn)
		#self.db = NNData(self.nlOut,1);
		#exit()
		print '======99======='
		self.W.mPrint()
		print '============='
		#	print self.B.mPrint()
		#exit()

	def forward(self, inData):
		self.outData = NNData(self.nlOut, inData.columns);
		
		#self.outData = self.W * inData + self.B;

		Wx = np.matmul(self.W.data,inData.data);
		print Wx;
		self.B.data[0,0] = 4;
		self.B.data[1,0] = 3;
		self.B.data[2,0] = 2;
		
		self.B.mPrint()
		self.outData.data = Wx + self.B.data;
		self.outData.mPrint()
		exit()
		return self.outData


