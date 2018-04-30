from NNLayer import NNLayer

class NNActivation(NNLayer):
	def __init__(self, network, name, nlOut):
		NNLayer.__init__(self, network, name, nlOut)
		if self.nlIn == self.nlOut:
			print "   > successful connection"
		else:
			print "   > Inconsistent Dimensions"
			exit()


class NNSigmoid(NNActivation):

	def forward(self, inData):
		self.outData = inData.sigmoid();
		return self.outData

	def backprop(self, dGlobal):
		y = self.outData.data;
		dGlobalNew = dGlobal * y * (1-y); #element by element mult since np.array 
		return dGlobalNew


class NNRelu(NNActivation):
			
	def forward(self, inData):
		self.outData = inData.relu();
		return self.outData

	def backprop(self, dGlobal):
		print '<< back propagating', self.name

		return dGlobal

		
