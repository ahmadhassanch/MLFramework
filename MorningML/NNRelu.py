from NNLayer import NNLayer

class NNRelu(NNLayer):
	def __init__(self, name, nlOut, nlIn):
		NNLayer.__init__(self, name, nlOut, nlIn)

	def forward(self):
		self.mPrint()
