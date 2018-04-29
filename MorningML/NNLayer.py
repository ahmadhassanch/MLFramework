from NNData import NNData

class NNLayer:

	def __init__(self, network, name, nlOut):
		self.name = name;
		self.network = network
		self.nlOut = nlOut

		if len(network.layers)>0:
			nlIn = network.layers[-1].nlOut
			#print 'inputs = ', nlIn 
			self.nlIn = nlIn
		else:
			self.nlIn = -1;

		network.addLayer(self);
		self.outData = 'UnInitialized OutData';
		print "Constructor", self.name


	def initWeights(self):
		print self.name, ": No Weights, biases"

	def forward(self, inData):
		print 'creating input Data', self.name
		
		
	def mPrint(self):
		print self.name, ' - ', self.nlOut, self.nlIn
