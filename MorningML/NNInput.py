from NNLayer import NNLayer

class NNInput(NNLayer):
	def __init__(self, network, name, nlOut):
		NNLayer.__init__(self, network, name, nlOut)

	def forward(self, inData):
		self.outData = inData;
		return self.outData
