from NNLayer import NNLayer

class NNInput(NNLayer):
	def __init__(self, name, nlOut, nlIn):
		NNLayer.__init__(self, name, nlOut, nlIn)

	def forward(self, inData):
		self.mPrint()
		self.outData = inData;
		return self.outData
