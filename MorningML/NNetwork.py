
class NNetwork:

	def __init__(self):
		self.name = "NNetwork"
		self.layers = [];
		self.numLayers = len(self.layers)

	def addLayer(self, layer):
		self.layers.append(layer);
		self.numLayers = len(self.layers)
		return layer.nlOut

	def initWeights(self):
		print '=================================================='
		print 'Doing Network Initialization of Weights and Biases'
		print '=================================================='
		for i in range(self.numLayers):
			self.layers[i].initWeights();

	def forward(self, inData):
		print '=================================================='
		print '>>>>>>>>>>>>>> Forward Network <<<<<<<<<<<<<<<<<<<'
		print '=================================================='
		outData = inData;
		for i in range(self.numLayers):
			print '\n===>>', self.layers[i].name, '<<==='
			outData = self.layers[i].forward(outData);
			outData.mPrint()

	def mPrint(self):
		print 'And I have', len(self.layers), 'layers'
		for i in range(self.numLayers):
			self.layers[i].mPrint();
