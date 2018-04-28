from NNLayer import NNLayer

class NNRelu(NNLayer):
	def __init__(self, network, name, nlOut):
		NNLayer.__init__(self, network, name, nlOut)

	def forward(self, inData):
		self.mPrint()
		self.outData = inData;
		return self.outData
