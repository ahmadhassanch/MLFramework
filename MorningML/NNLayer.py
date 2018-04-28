from NNData import NNData

class NNLayer:

	def __init__(self, name, nlOut, nlIn):
		self.name = name;
		self.nlOut = nlOut
		self.nlIn = nlIn

	def forward(self):
		print 'Forward Prop Layer: ', self.name

	def createInputData(self):
		print 'creating input Data', self.name
		
	def mPrint(self):
		print 'I am Layer: ', self.name, ' - ', self.nlOut, self.nlIn
