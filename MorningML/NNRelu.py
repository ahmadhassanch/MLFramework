from NNLayer import NNLayer

class NNRelu(NNLayer):
	def __init__(self, name, nlOut, nlIn):
		NNLayer.__init__(self, name, nlOut, nlIn)

	def forward(self, inData):
		self.mPrint()
		self.outData = inData;
		return self.outData
