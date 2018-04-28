from NNData import NNData

class NNLayer:

	def __init__(self, name, nlOut, nlIn):
		self.name = name;
		self.nlOut = nlOut
		self.nlIn = nlIn
		self.outData = 'UnInitialized OutData';

	def initWeights(self):
		print self.name, ": No Weights, biases"

	def forward(self, inData):
		print 'creating input Data', self.name
		self.outData = inData;
		return self.outData
		
	def mPrint(self):
		print 'I am Layer: ', self.name, ' - ', self.nlOut, self.nlIn
