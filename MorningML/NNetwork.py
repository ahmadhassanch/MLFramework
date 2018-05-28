import numpy as np

class NNetwork:

	def __init__(self):
		self.name = "NNetwork"
		self.layers = [];
		self.layerDict = {};

	def addLayer(self, layer):
		self.layers.append(layer);
		self.layerDict[layer.name] =  layer;
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
		self.yRefData = yRefData
		yHat = self.layers[-1].outData.data;
		y = yRefData.data;
		#print y
		#print yRef
		#print np.log(1-yHat+1e-10)
		#exit()
		loss = - (y * np.log(yHat) + (1-y) * np.log(1-yHat+1e-10))
		#exit()
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
		#exit()
		dA = -y / yHat + (1-y) / (1-yHat+1e-10)
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

	def debugInfo(self):
		for layer in self.layers:
			layer.debugInfo();
		self.yRefData.mPrintSTD()

	def gradientCheck(self, refDataX, refDataY):
		epsilon = 1e-3;
		for layer in self.layers:

			if(layer.type != "InnerProduct"): continue;
			print ">>>>>>>>>>>", layer.name
			W = layer.W.data;
			dJW = layer.dW.data;
			for i in range(W.shape[0]):
				for j in range(W.shape[1]):
					print i, j
					w = W[i, j];

					# Run Foward twice for computing derivative (+/-)
					W[i, j] = w + epsilon;

					self.forward(refDataX);
					y, yHat, loss, Jp = self.computeLoss(refDataY)

					W[i, j] = w - epsilon;
					self.forward(refDataX);
					y, yHat, loss, Jm = self.computeLoss(refDataY)

					# Computing numerical derivate
					dJw1 = (Jp - Jm) / (2*epsilon)

					# Restore weight and Forward/Backward and get dW
					W[i, j] = w
					self.forward(refDataX);
					y, yHat, loss, Jm = self.computeLoss(refDataY)
					self.backprop(y, yHat);
											
					dJw2 = dJW[i, j]
					diff = abs(dJw2 - dJw1);
					if(diff > 1e-10): 
						print "Error ", dJw1, dJw2, diff; 
						exit()



	def mPrint(self):
		print 'And I have', len(self.layers), 'layers'
		for i in range(self.numLayers):
			self.layers[i].mPrint();
