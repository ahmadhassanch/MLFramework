import numpy as np

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
		#print '=================================================='
		#print '>>>>>>>>>>>>>> Forward Network <<<<<<<<<<<<<<<<<<<'
		#print '=================================================='
		outData = inData;
		for layer in self.layers:
			#print '\n===>>', layer.name, '<<=== ', (layer.nlOut, layer.nlIn)
			#print outData.data.shape
			outData = layer.forward(outData);
			#outData.mPrint()
		#exit()
		return outData.data

	def computeLoss(self, yRefData):
		#print '\n=========== Computing LOSS ============'
		yHat = self.layers[-1].outData.data;
		y = yRefData.data;
		#print y
		#print yRef
		loss = - (y * np.log(yHat) + (1-y) * np.log(1-yHat))
		#print loss
		m = y.shape[1];
		#print m
		J= 1.0/m*np.sum(loss);
		#print "J =================================================== J = ", J
		#print loss.shape 
		return y, yHat, loss, J
		

	def backprop(self, y, yHat):
		#print '=================================================='
		#print '>>>>>>>>>>>>>> Backward Network <<<<<<<<<<<<<<<<<<<'
		#print '=================================================='

		dA = -y / yHat + (1-y) / (1-yHat)
		#print dA
		dGlobal = dA;
		for layer in reversed(self.layers):
			#print '\n<<===>>', layer.name, '<<===>>'
			dGlobal = layer.backprop(dGlobal);
			#print dGlobal

			#outData.mPrint()

	def gradientDescent(self, alpha):
		for layer in self.layers:
			layer.gradientDescent(alpha);

	def saveWeights(self):
		for layer in self.layers:
			layer.saveWeights();


	def loadWeights(self):
		for layer in self.layers:
			layer.loadWeights();

	def restorePivot(self):
		for layer in self.layers:
			layer.restorePivot();


	def mPrint(self):
		print 'And I have', len(self.layers), 'layers'
		for i in range(self.numLayers):
			self.layers[i].mPrint();
