
class NNetwork:

	def __init__(self):
		self.name = "NNetwork"
		self.layers = [];
		self.numLayers = len(self.layers)

	def addLayer(self, layer):
		self.layers.append(layer);
		self.numLayers = len(self.layers)
		return layer.nlOut

	def forward(self, inData):
		print 'Doing Network Forward'
		outData = inData;
		for i in range(self.numLayers):
			outData = self.layers[i].forward(outData);



	def mPrint(self):
		print 'I am Network', self.name;

		print 'And I have', len(self.layers), 'layers'
		for i in range(self.numLayers):
			self.layers[i].mPrint();
