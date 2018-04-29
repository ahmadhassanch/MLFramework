
class NNetwork:

	def __init__(self):
		self.name = "NNetwork"
		self.layers = [];

	def addLayer(self, layer):
		self.layers.append(layer);
		return layer.nlOut

	def initWeights(self):
		print '=================================================='
		print 'Doing Network Initialization of Weights and Biases'
		print '=================================================='
		for layer in self.layers:
			layer.initWeights();

	def forward(self, inData):
		print '=================================================='
		print '>>>>>>>>>>>>>> Forward Network <<<<<<<<<<<<<<<<<<<'
		print '=================================================='
		outData = inData;
		for layer in self.layers:
			print '\n===>>', layer.name, '<<==='
			outData = layer.forward(outData);
			outData.mPrint()

	def computeLoss(self, refY):
		print '\n=========== Computing LOSS ============'

	def backprop(self, inData):
		print '=================================================='
		print '>>>>>>>>>>>>>> Forward Network <<<<<<<<<<<<<<<<<<<'
		print '=================================================='
		outData = inData;
		for layer in reversed(self.layers):
			print '\n<<===>>', layer.name, '<<===>>'
			layer.backprop(outData);
			#outData.mPrint()

	def mPrint(self):
		print 'And I have', len(self.layers), 'layers'
		for i in range(self.numLayers):
			self.layers[i].mPrint();
