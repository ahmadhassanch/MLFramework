from NNData import NNData

class NNLayer:

	def __init__(self, network, name, nlOut):
		self.name = name;
		self.network = network
		self.nlOut = nlOut
		self.type = "BaseLayer"

		if len(network.layers)>0:
			nlIn = network.layers[-1].nlOut
			self.nlIn = nlIn
		else:
			self.nlIn = -1;

		network.addLayer(self);
		self.outData = 'UnInitialized OutData';
		print "Constructor Called For", self.name


	def initWeights(self):
		#print self.name, ": No Weights, biases"
		pass

	def saveWeights(self):
		#print self.name, ": No Weights, biases"
		pass

	def loadWeights(self):
		#print self.name, ": No Weights, biases"
		pass

	def restorePivot(self):
		pass

	def debugInfo(self):
		pass


	def forward(self, inData):
		#print 'creating input Data', self.name
		pass
		
	def backprop(self, dL):
		#print '<< back propagating', self.name
		return dL

	def gradientDescent(self, alpha):
		pass
		#print '<< update Gradients', self.name
		
	def mPrint(self):
		print self.name, ' - ', self.nlOut, self.nlIn


class NNInput(NNLayer):
	def __init__(self, network, name, nlOut):
		NNLayer.__init__(self, network, name, nlOut)

	def forward(self, inData):
		self.outData = inData;
		return self.outData
