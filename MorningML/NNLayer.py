from NNData import NNData

class NNLayer:

	def __init__(self, network, name, nlOut):
		self.name = name;
		self.network = network
		self.nlOut = nlOut
		network.addLayer(self);

		if len(network.layers)>0:
			nlIn = network.layers[-1].nlOut
			print 'inputs = ', nlIn 
			self.nlIn = nlIn
		else:
			self.nlIn = -1;
		self.outData = 'UnInitialized OutData';


	def initWeights(self):
		print self.name, ": No Weights, biases"

	def forward(self, inData):
		print 'creating input Data', self.name
		self.outData = inData;
		return self.outData
		
	def mPrint(self):
		print self.name, ' - ', self.nlOut, self.nlIn
